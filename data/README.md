# Dataset

![../figs/fig_finee_case.png](figs/fig_finee_case.png)

| **Key**          | **Value**   |
|------------------|-------------|
| Dataset Name     | DocFEE       |
| Size             | 19,044      |
| Event Type       | 9           |
| Argument Type    | 38          |
| Event Range      | 960.06      |
| Event Count      | 1.86        |
| Document Length  | 2277.25     |

| Event Type                | Arguments                                                                                   |
|---------------------------|---------------------------------------------------------------------------------------------|
| Bankruptcy Liquidation    | Company Name, Industry, Announcement Date, Court of Acceptance, Judgment Date               |
| Major Safety Incident     | Number of Casualties, Company Name, Announcement Date, Other Impacts, Loss Amount           |
| Shareholder Reduction     | Reduction Start Date, Shareholder, Reduction Amount                                         |
| Equity Pledge             | Receiver, Pledge Start Date, Pledgor, Pledge End Date, Pledge Amount                         |
| Shareholder Increase      | Increase Start Date, Shareholder, Increase Amount                                           |
| Equity Freeze             | Freeze Start Date, Freeze End Date, Freeze Amount, Frozen Shareholder                       |
| Senior Executive Death    | Company Name, Death/Missing Date, Age at Death, Executive, Position                         |
| Major Asset Loss          | Company Name, Announcement Date, Other Losses, Loss Amount                                  |
| Major External Compensation | Company Name, Announcement Date, Compensation Recipient, Compensation Amount               |

Run `unzip DocFEE_dataset.zip` to get the train and test set of DocFEE.
