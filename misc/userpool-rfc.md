# Userpool for Crawlee

> This document is a proposal for a new `UserPool` class in Crawlee, which aims to simplify resource management by unifying the handling of browser and session resources.
> The actual implementation of this RFC is available in the [feat/user-pool]() branch of the Crawlee repository.

Resource management is currently done in multiple places (`BrowserPool`, `SessionPool`), which leads to complexity and potential resource conflicts. This RFC proposes a unified `UserPool` and `ResourceOwner` classes to streamline resource management and improve efficiency.

## 1. Motivation

> For more details on the motivation, see the [UserPool design and thoughts](https://www.notion.so/apify/UserPool-design-and-thoughts-1c7f39950a2280118608daeb0950d76c) document.

## 2. `ResourceOwner` class

The core of this proposal is the `ResourceOwner` class, which is responsible for **owning** arbitrary resources. It can be referenced by `id`.

```typescript
export abstract class ResourceOwner<ResourceType> {
    // Arbitrary ID to identify the resource owner.
    // Multiple user instances can share the same ID.
    public id: string = '';

    // Runs the passed task with the managed resource as an argument.
    public abstract runTask<T, TaskReturnType extends Promise<T> | T>(
        task: (resource: ResourceType) => TaskReturnType,
    ): Promise<T>;

    // `ResourceOwner` subclasses can decide to lock the resource while it's being used.
    // This is useful for resources that are stateful (e.g. browser pages).
    public abstract isIdle(): boolean;
}
```

## 3. `UserPool` class

The `UserPool` class is a **container for `ResourceOwner`** instances. It allows users to manage `ResourceOwner` instances and provides methods to acquire and release resources.

```typescript
export class UserPool<ResourceType> {
    constructor(private users: ResourceOwner<ResourceType>[] = []) {}

    public getUser(filter?: { id?: string }): ResourceOwner<ResourceType> | undefined;
    public hasIdleUsers(filter?: { id?: string }): boolean;
}
```

## 4. Crawler implementation

The `UserPool` class can be used to manage resources in a unified way. Here's a simplified example of how to use it:

### BrowserCrawler Example

```typescript
class BrowserUser extends ResourceOwner<Browser> {
    ...
    public async runTask(task) {
        this._isIdle = false;
        const result = await task(this.browser);
        this._isIdle = true;
        return result;
    }

    public isIdle(): boolean {
        return true; // Mutiple tasks can use the browser concurrently.
    }
    ...
}

class BasicCrawler<ResourceType> {
    ...
    userPool: UserPool<ResourceType>;
    ...
}

class BrowserCrawler extends BasicCrawler<Browser> {
    ...
    async requestHandler(context: RequestHandlerContext): Promise<void> {
        const user = await this.userPool.getUser();
        if (user) {
            // Use the resource
            await user.runTask(async (browser) => {
                const page = await browser.newPage();
                await page.goto(context.request.url);

                await this.userRequestHandler({
                    ...context,
                    browser,
                });

                await page.close();
            });
        }
    }
    ...
}
```
