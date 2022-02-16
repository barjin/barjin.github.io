# Competition Analysis - Declarative Web Automation Framework

Since the web automation industry is growing at a rapid rate, it is crucial to match the competition. 
The following text offers a comparison of available competing products with the proposed project on various qualities. 

The comparison is made on the following points:
- **Ease of use** - The project should allow the user to create web automations using a comprehensive UI.
- **Workflow resilience** - The workflow format allows to define conditions for certain actions, letting the interpreter to choose based on the current state. This helps to mitigate runtime exceptions from getting into an unexpected state, caused e.g. by an A/B test experiment).
- **Open format** - The definition of the format is open and allows for simple third-party application development.

|   | Ease of Use | Resilience | Open format | Language support |
|---|---|---|---|---|
| [Selenium Web Driver](https://www.selenium.dev/), [Playwright](https://playwright.dev/), [Puppeteer](https://github.com/puppeteer/puppeteer) | ❌ Programmming required | ❌ Not inherent <sup>1</sup> | 🟨 Source code | JS, Java, C#, Python... 
| [Dexi.io](https://www.dexi.io/) | ✅ WYSIWYG recorder (web app) | 🟨 Not ensured <sup>2</sup> | ❌ JSON-based proprietary format | runnable by a proprietary SW only
| [Browse.ai](https://www.browse.ai/) | ✅ WYSIWYG recorder (Chrome extension) | ❌ Not possible | ❌ Closed-source format (vendor lock-in) | no exportable file (only executable via platform)
| **WBR (this SW)** | ✅ WYSIWYG editor (web app) | ✅ Rooted in the format design | ✅ JSON-based open format | language-agnostic
 
____

<sup>1</sup> The programmer has to account for edge cases using their own code.\
<sup>2</sup> The author has to explain the workflow split points manually.