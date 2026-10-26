# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: senate.spec.ts >> ribbon displays all margin-ordered races, coincident markers, and scenario-specific tipping frequencies
- Location: tests/site/senate.spec.ts:205:5

# Error details

```
Error: expect(received).toEqual(expected) // deep equality

- Expected  - 8
+ Received  + 8

  Array [
    "2026:RI:senate:regular",
    "2026:MA:senate:regular",
+   "2026:OR:senate:regular",
    "2026:DE:senate:regular",
-   "2026:OR:senate:regular",
    "2026:CO:senate:regular",
    "2026:NJ:senate:regular",
    "2026:VA:senate:regular",
    "2026:NM:senate:regular",
    "2026:IL:senate:regular",
    "2026:GA:senate:regular",
-   "2026:NH:senate:regular",
    "2026:NC:senate:regular",
+   "2026:NH:senate:regular",
    "2026:MN:senate:regular",
-   "2026:ME:senate:regular",
    "2026:MI:senate:regular",
+   "2026:ME:senate:regular",
    "2026:OH:senate:special",
    "2026:TX:senate:regular",
    "2026:AK:senate:regular",
    "2026:IA:senate:regular",
    "2026:KS:senate:regular",
-   "2026:ID:senate:regular",
-   "2026:MT:senate:regular",
+   "2026:SC:senate:regular",
    "2026:NE:senate:regular",
-   "2026:SC:senate:regular",
    "2026:FL:senate:special",
+   "2026:MT:senate:regular",
+   "2026:ID:senate:regular",
    "2026:LA:senate:regular",
+   "2026:MS:senate:regular",
    "2026:AR:senate:regular",
-   "2026:MS:senate:regular",
+   "2026:SD:senate:regular",
    "2026:KY:senate:regular",
    "2026:AL:senate:regular",
-   "2026:SD:senate:regular",
    "2026:OK:senate:regular",
    "2026:TN:senate:regular",
    "2026:WV:senate:regular",
    "2026:WY:senate:regular",
  ]
```

# Page snapshot

```yaml
- generic [ref=e1]:
  - generic [ref=e2]:
    - link "Skip to the races" [ref=e3] [cursor=pointer]:
      - /url: "#races"
    - banner [ref=e4]:
      - link "Ballotmancy home" [ref=e5] [cursor=pointer]:
        - /url: /
        - generic [aria-hidden] [ref=e6]: b.
        - text: ballotmancy
      - navigation "Main navigation" [ref=e7]:
        - link "The races" [ref=e8] [cursor=pointer]:
          - /url: "#races"
        - link "How to read this" [ref=e9] [cursor=pointer]:
          - /url: "#methodology"
        - generic [ref=e10]: Research preview
    - generic [ref=e12]:
      - tablist "Chamber" [ref=e13]:
        - tab "Senate" [selected] [ref=e14] [cursor=pointer]
        - tab "House" [ref=e15] [cursor=pointer]
      - generic [ref=e16]: Two chambers. Your perspective.
    - tabpanel "Senate" [ref=e17]:
      - main [ref=e18]:
        - generic [ref=e19]:
          - generic [ref=e20]:
            - generic [ref=e21]: 2026 United States Senate · 35 elections
            - heading "The Senate, in perspective." [level=1] [ref=e22]: The Senate,in perspective.
            - paragraph [ref=e23]: Choose the model’s ingredients. Explore the races and chamber together.A closer look at the races that could shape the chamber, with visible uncertainty.
            - group [ref=e24]:
              - generic "Input and publication times" [ref=e25] [cursor=pointer]
          - generic [ref=e26]:
            - text: THE RESEARCH EDITION
            - strong [ref=e27]: October 26, 2026
            - text: Inputs dated through October 26, 2026
        - group [ref=e28]:
          - generic "Build your perspective Standard polling Collapse controls" [active] [ref=e29] [cursor=pointer]:
            - generic [ref=e30]:
              - generic [ref=e31]: Build your perspective
              - generic [ref=e32]: Standard polling
            - generic [ref=e33]:
              - generic [ref=e34]: Collapse controls
              - generic [aria-hidden] [ref=e35]: ⌄
          - generic [ref=e36]:
            - generic [ref=e37]:
              - generic [ref=e38]:
                - generic [ref=e39]: Customize this view
                - generic [ref=e40]: A stack is your combination of model choices. Select saved results or inspect experimental alternatives; experimental choices may need a new calculation before they can be shown.
              - generic [ref=e41]:
                - generic [ref=e42] [cursor=pointer]:
                  - checkbox "Compare default" [ref=e43]
                  - text: Compare default
                - button "Reset stack" [ref=e44] [cursor=pointer]
                - button "Share view ↗" [ref=e45] [cursor=pointer]
            - generic [ref=e46]:
              - generic [ref=e47]:
                - generic [ref=e48]: Election year
                - group "Election year" [ref=e49]:
                  - button "2026" [pressed] [ref=e50] [cursor=pointer]
                  - button "2022" [ref=e51] [cursor=pointer]
                  - button "2018" [ref=e52] [cursor=pointer]
              - generic [ref=e53]:
                - generic [ref=e54]: Forecast date
                - combobox "Forecast date" [ref=e55]:
                  - option "October 20, 2026"
                  - option "October 26, 2026" [selected]
                - paragraph [ref=e56]: Choose an available forecast date. Each date has its own saved component combinations; changing a component can switch to its saved edition.
            - generic [ref=e57]:
              - generic [ref=e58]:
                - strong [ref=e59]: Explore individual model components
                - paragraph [ref=e60]: A 2018 study separates more of the model’s ingredients. It covers 30 Democratic–Republican races and has its own race scorecard.
              - button "Explore the 2018 component study" [ref=e61] [cursor=pointer]
            - generic [ref=e62]:
              - generic [ref=e63]: Preserved presets · ready-made combinations
              - group "Preserved forecast presets" [ref=e64]:
                - button "2026 Basic electoral history v5" [ref=e65] [cursor=pointer]:
                  - generic [ref=e66]: "2026"
                  - text: Basic electoral history v5
                - button "2026 Best guess with October 26 polling" [pressed] [ref=e67] [cursor=pointer]:
                  - generic [ref=e68]: "2026"
                  - text: Best guess with October 26 polling
            - status [ref=e69]:
              - strong [ref=e70]: Ready
              - paragraph [ref=e71]:
                - generic [ref=e72]: "Displaying:"
                - strong [ref=e73]: Standard polling
                - text: · Research preview · October 26, 2026 (2026-10-26).
            - region [ref=e74]:
              - generic [ref=e75]:
                - heading "The full model, slice by slice" [level=3] [ref=e76]
                - paragraph [ref=e77]: Choose the ingredients for your requested setup. Results change when a matching saved forecast loads. Unfinished slices stay visible as the model develops.
              - generic [ref=e78]:
                - region [ref=e79]:
                  - generic [ref=e80]:
                    - generic [ref=e81]:
                      - checkbox "Include population & geography" [disabled] [ref=e82]
                      - heading "Population & geography" [level=4] [ref=e83]
                      - generic [ref=e84]: Not connected yet
                    - paragraph [ref=e85]: Defines who is represented, where, and for which dates.
                  - paragraph [ref=e86]: A population model is not connected yet. These forecasts use a fixed list of contests.
                - region [ref=e87]:
                  - generic [ref=e88]:
                    - generic [ref=e89]:
                      - checkbox "Include community composition" [disabled] [ref=e90]
                      - heading "Community composition" [level=4] [ref=e91]
                      - generic [ref=e92]: Not connected yet
                    - paragraph [ref=e93]: Models measured groups and uncertainty about how they overlap.
                  - paragraph [ref=e94]: Community composition is not estimated in these forecasts.
                - region [ref=e95]:
                  - generic [ref=e96]:
                    - generic [ref=e97]:
                      - generic [aria-hidden] [ref=e98]: ✓
                      - heading "Electoral history" [level=4] [ref=e99]
                      - generic [ref=e100]: Always included
                    - paragraph [ref=e101]: Sets a starting estimate from earlier elections and candidate context.
                  - generic [ref=e103]:
                    - generic [ref=e104]: Base model
                    - combobox "Base model" [ref=e105]:
                      - option "Electoral-history starting estimate" [selected]
                      - option "Recent calendar history"
                      - option "Refreshed history"
                      - option "Learned presidential slope"
                      - option "Relative lean with presidential context"
                      - option "Relative state and national lean"
                    - paragraph [ref=e106]: Starts from electoral history, combining information into a race-level estimate before adding any selected polls. This is the saved fundamentals model.
                - region [ref=e107]:
                  - generic [ref=e108]:
                    - generic [ref=e109]:
                      - generic [aria-hidden] [ref=e110]: ✓
                      - heading "National environment" [level=4] [ref=e111]
                      - generic [ref=e112]: Always included
                    - paragraph [ref=e113]: Sets uncertainty and how strongly races can move together.
                  - generic [ref=e115]:
                    - generic [ref=e116]: Uncertainty
                    - combobox "Uncertainty" [ref=e117]:
                      - option "Standard uncertainty" [selected]
                      - option "Polling-error sizes × 2/3"
                      - option "Polling-error sizes × 1.5"
                      - option "Shared election-wide error size × 0.5"
                      - option "Shared election-wide error size × 2"
                    - paragraph [ref=e118]: Uses the saved default assumptions about how much results can vary and how races can move together.
                - region [ref=e119]:
                  - generic [ref=e120]:
                    - generic [ref=e121]:
                      - generic [aria-hidden] [ref=e122]: ✓
                      - heading "Polls & surveys" [level=4] [ref=e123]
                      - generic [ref=e124]: Included
                    - paragraph [ref=e125]: Adds polling evidence while allowing for survey error.
                  - generic [ref=e127]:
                    - generic [ref=e128] [cursor=pointer]:
                      - checkbox "Include polling evidence" [checked] [ref=e129]
                      - text: Include polling evidence
                    - paragraph [ref=e130]: Standard polling
                    - paragraph [ref=e131]: "Adds polls using a fixed bell-shaped (Gaussian) model for measurement error: polls are evidence with uncertainty, not exact election results. Only saved combinations can be displayed."
                - region [ref=e132]:
                  - generic [ref=e133]:
                    - generic [ref=e134]:
                      - checkbox "Include turnout & participation" [disabled] [ref=e135]
                      - heading "Turnout & participation" [level=4] [ref=e136]
                      - generic [ref=e137]: Not connected yet
                    - paragraph [ref=e138]: Models who votes and how turnout varies.
                  - paragraph [ref=e139]: A turnout model is not connected yet. These forecasts do not predict voter counts.
                - region [ref=e140]:
                  - generic [ref=e141]:
                    - generic [ref=e142]:
                      - generic [aria-hidden] [ref=e143]: ✓
                      - heading "Candidate choice" [level=4] [ref=e144]
                      - generic [ref=e145]: Built into this model
                    - paragraph [ref=e146]: Models choices among candidates and ballot options.
                  - paragraph [ref=e147]: The candidate options are fixed in each saved forecast. There is no separate candidate-choice control.
                - region [ref=e148]:
                  - generic [ref=e149]:
                    - generic [ref=e150]:
                      - generic [aria-hidden] [ref=e151]: ✓
                      - heading "Vote counts & election rules" [level=4] [ref=e152]
                      - generic [ref=e153]: Built into this model
                    - paragraph [ref=e154]: Turns voting choices into counts, winners and chamber outcomes.
                  - paragraph [ref=e155]: Chamber calculations and tie rules are fixed in each saved Senate result. Total vote counts are not predicted, and there is no separate control to change these rules.
              - paragraph [ref=e156]: “Always included” means this part is required by these saved models. “Built into this model” means it is present but has no separate control. Disabled choices cannot change this result. These choices belong to one model with shared assumptions; they are not percentages to blend.
            - generic [ref=e157]:
              - generic [ref=e158]:
                - generic [ref=e159]: Comparison baseline
                - combobox "Comparison baseline" [ref=e160]:
                  - option "2026 default" [selected]
                  - 'option "Polling error: shared election-wide error size × 2"'
                  - 'option "Polling error: shared election-wide error size × 0.5"'
                  - 'option "Polling error: all error sizes × 1.5"'
                  - 'option "Polling error: all error sizes × 2/3"'
                - paragraph [ref=e161]: A baseline is the reference result you compare against. Choices must predict the same election and type of outcome on the same date. The historical scorecard uses this reference; the checkbox adds comparison details below.
              - group [ref=e162]:
                - generic "Saved views (0) · This browser" [ref=e163] [cursor=pointer]:
                  - text: Saved views
                  - generic [ref=e164]: (0) · This browser
            - group [ref=e165]:
              - generic "Configuration, evidence & data dates" [ref=e166] [cursor=pointer]
        - generic [ref=e167]:
          - generic [ref=e168]:
            - generic [ref=e169]: Standard polling · 40,000 saved simulations
            - generic [ref=e170]:
              - generic [ref=e171]: Who joins each party?
              - combobox "Who joins each party?" [ref=e172]:
                - option "Independents unaligned" [selected]
                - option "Independents work with Democrats"
                - option "Independents work with Republicans"
          - paragraph [ref=e173]: Each simulation is one possible set of election results. A caucus is a party’s group in the Senate, including independents who work with it. Choose those alliances above; candidate party labels stay the same.
          - region [ref=e174]:
            - generic [ref=e175]:
              - generic [ref=e176]:
                - heading "The path to a majority" [level=2] [ref=e177]
                - paragraph [ref=e178]: 35 elections, ordered by the projected vote-share lead between the two modeled contenders.
              - generic [ref=e179]:
                - generic [ref=e180]: D
                - generic [ref=e181]: R
                - text: Projected control thresholds
            - generic [aria-hidden] [ref=e182]:
              - generic [ref=e183]: Democratic / independent advantage
              - generic [ref=e184]: Republican advantage
            - region "Senate race ribbon, ordered by projected margin. Scroll horizontally for all 35 elections." [ref=e185]:
              - generic [ref=e186]:
                - generic [ref=e187]:
                  - strong [ref=e188]: "34"
                  - generic [ref=e189]: D caucusnot up this year
                - button "Rhode Island, regular election, Safe, projected margin D +31.4 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e190] [cursor=pointer]:
                  - generic [ref=e191]: RI
                  - generic [ref=e192]: ’26
                - button "Massachusetts, regular election, Safe, projected margin D +30.8 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e193] [cursor=pointer]:
                  - generic [ref=e194]: MA
                  - generic [ref=e195]: ’26
                - button "Oregon, regular election, Safe, projected margin D +28.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e196] [cursor=pointer]:
                  - generic [ref=e197]: OR
                  - generic [ref=e198]: ’26
                - button "Delaware, regular election, Safe, projected margin D +28.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e199] [cursor=pointer]:
                  - generic [ref=e200]: DE
                  - generic [ref=e201]: ’26
                - button "Colorado, regular election, Safe, projected margin D +25.4 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations." [ref=e202] [cursor=pointer]:
                  - generic [ref=e203]: CO
                  - generic [ref=e204]: ’26
                - button "New Jersey, regular election, Safe, projected margin D +20.5 pp. At the D control threshold in 0.5% of simulations. At the R control threshold in 0.5% of simulations." [ref=e205] [cursor=pointer]:
                  - generic [ref=e206]: NJ
                  - generic [ref=e207]: ’26
                - button "Virginia, regular election, Safe, projected margin D +20.4 pp. At the D control threshold in 0.5% of simulations. At the R control threshold in 0.5% of simulations." [ref=e208] [cursor=pointer]:
                  - generic [ref=e209]: VA
                  - generic [ref=e210]: ’26
                - button "New Mexico, regular election, Safe, projected margin D +17.7 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e211] [cursor=pointer]:
                  - generic [ref=e212]: NM
                  - generic [ref=e213]: ’26
                - button "Illinois, regular election, Safe, projected margin D +16.5 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e214] [cursor=pointer]:
                  - generic [ref=e215]: IL
                  - generic [ref=e216]: ’26
                - button "Georgia, regular election, Safe, projected margin D +8.2 pp. At the D control threshold in 1.6% of simulations. At the R control threshold in 1.7% of simulations." [ref=e217] [cursor=pointer]:
                  - generic [ref=e218]: GA
                  - generic [ref=e219]: ’26
                - button "North Carolina, regular election, Safe, projected margin D +8.1 pp. At the D control threshold in 1.3% of simulations. At the R control threshold in 1.4% of simulations." [ref=e220] [cursor=pointer]:
                  - generic [ref=e221]: NC
                  - generic [ref=e222]: ’26
                - button "New Hampshire, regular election, Safe, projected margin D +7.7 pp. At the D control threshold in 2.1% of simulations. At the R control threshold in 2.1% of simulations." [ref=e223] [cursor=pointer]:
                  - generic [ref=e224]: NH
                  - generic [ref=e225]: ’26
                - button "Minnesota, regular election, Likely, projected margin D +4.8 pp. At the D control threshold in 5.2% of simulations. At the R control threshold in 5.2% of simulations." [ref=e226] [cursor=pointer]:
                  - generic [ref=e227]: MN
                  - generic [ref=e228]: ’26
                - button "Michigan, regular election, Likely, projected margin D +2.6 pp. At the D control threshold in 8.9% of simulations. At the R control threshold in 8.6% of simulations." [ref=e229] [cursor=pointer]:
                  - generic [ref=e230]: MI
                  - generic [ref=e231]: ’26
                - button "Maine, regular election, Likely, projected margin D +1.8 pp. At the D control threshold in 9.9% of simulations. At the R control threshold in 9.3% of simulations." [ref=e232] [cursor=pointer]:
                  - generic [ref=e233]: ME
                  - generic [ref=e234]: ’26
                - button "Ohio, special election, Tossup, projected margin R +0.3 pp. At the D control threshold in 13.1% of simulations. At the R control threshold in 11.9% of simulations." [pressed] [ref=e235] [cursor=pointer]:
                  - generic [ref=e236]: OH
                  - generic [ref=e237]: SP
                - button "Texas, regular election, Tossup, projected margin R +1.1 pp, projected D control threshold, projected R control threshold. At the D control threshold in 13.0% of simulations. At the R control threshold in 11.5% of simulations." [ref=e238] [cursor=pointer]:
                  - generic [ref=e240]: D51 / R50 · TX
                  - generic [ref=e241]: TX
                  - generic [ref=e242]: ’26
                - button "Alaska, regular election, Likely, projected margin R +2.1 pp. At the D control threshold in 10.1% of simulations. At the R control threshold in 8.8% of simulations." [ref=e243] [cursor=pointer]:
                  - generic [ref=e244]: AK
                  - generic [ref=e245]: ’26
                - button "Iowa, regular election, Likely, projected margin R +2.5 pp. At the D control threshold in 11.4% of simulations. At the R control threshold in 9.8% of simulations." [ref=e246] [cursor=pointer]:
                  - generic [ref=e247]: IA
                  - generic [ref=e248]: ’26
                - button "Kansas, regular election, Likely, projected margin R +3.7 pp. At the D control threshold in 8.7% of simulations. At the R control threshold in 7.4% of simulations." [ref=e249] [cursor=pointer]:
                  - generic [ref=e250]: KS
                  - generic [ref=e251]: ’26
                - button "South Carolina, regular election, Likely, projected margin R +5.4 pp. At the D control threshold in 6.0% of simulations. At the R control threshold in 5.0% of simulations." [ref=e252] [cursor=pointer]:
                  - generic [ref=e253]: SC
                  - generic [ref=e254]: ’26
                - button "Nebraska, regular election, Likely, projected margin R +5.7 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 4.7% of simulations." [ref=e255] [cursor=pointer]:
                  - generic [ref=e256]: NE
                  - generic [ref=e257]: ’26
                - button "Florida, special election, Safe, projected margin R +8.1 pp. At the D control threshold in 2.3% of simulations. At the R control threshold in 1.8% of simulations." [ref=e258] [cursor=pointer]:
                  - generic [ref=e259]: FL
                  - generic [ref=e260]: SP
                - button "Montana, regular election, Safe, projected margin R +8.3 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.4% of simulations." [ref=e261] [cursor=pointer]:
                  - generic [ref=e262]: MT
                  - generic [ref=e263]: ’26
                - button "Idaho, regular election, Likely, projected margin R +8.8 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.8% of simulations." [ref=e264] [cursor=pointer]:
                  - generic [ref=e265]: ID
                  - generic [ref=e266]: ’26
                - button "Louisiana, regular election, Likely, projected margin R +12.0 pp. At the D control threshold in 2.3% of simulations. At the R control threshold in 2.0% of simulations." [ref=e267] [cursor=pointer]:
                  - generic [ref=e268]: LA
                  - generic [ref=e269]: ’26
                - button "Mississippi, regular election, Safe, projected margin R +13.4 pp. At the D control threshold in 1.1% of simulations. At the R control threshold in 0.9% of simulations." [ref=e270] [cursor=pointer]:
                  - generic [ref=e271]: MS
                  - generic [ref=e272]: ’26
                - button "Arkansas, regular election, Safe, projected margin R +17.1 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.2% of simulations." [ref=e273] [cursor=pointer]:
                  - generic [ref=e274]: AR
                  - generic [ref=e275]: ’26
                - button "South Dakota, regular election, Safe, projected margin R +17.9 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e276] [cursor=pointer]:
                  - generic [ref=e277]: SD
                  - generic [ref=e278]: ’26
                - button "Kentucky, regular election, Safe, projected margin R +21.0 pp. At the D control threshold in 0.6% of simulations. At the R control threshold in 0.5% of simulations." [ref=e279] [cursor=pointer]:
                  - generic [ref=e280]: KY
                  - generic [ref=e281]: ’26
                - button "Alabama, regular election, Safe, projected margin R +21.5 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e282] [cursor=pointer]:
                  - generic [ref=e283]: AL
                  - generic [ref=e284]: ’26
                - button "Oklahoma, regular election, Safe, projected margin R +25.1 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.3% of simulations." [ref=e285] [cursor=pointer]:
                  - generic [ref=e286]: OK
                  - generic [ref=e287]: ’26
                - button "Tennessee, regular election, Safe, projected margin R +26.2 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e288] [cursor=pointer]:
                  - generic [ref=e289]: TN
                  - generic [ref=e290]: ’26
                - button "West Virginia, regular election, Safe, projected margin R +37.2 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e291] [cursor=pointer]:
                  - generic [ref=e292]: WV
                  - generic [ref=e293]: ’26
                - button "Wyoming, regular election, Safe, projected margin R +37.9 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.0% of simulations." [ref=e294] [cursor=pointer]:
                  - generic [ref=e295]: WY
                  - generic [ref=e296]: ’26
                - generic [ref=e297]:
                  - strong [ref=e298]: "31"
                  - generic [ref=e299]: R caucusnot up this year
            - paragraph [ref=e300]: Read inward from either party’s end. Its marker identifies the last race needed for control in this projected order, after seats not up for election and races guaranteed to join that party’s group. D = Democratic; R = Republican. “SP” marks a special election to fill an unfinished term.
            - generic [ref=e301]:
              - generic [ref=e302]:
                - generic [ref=e303]: D
                - generic [ref=e304]:
                  - strong [ref=e305]:
                    - button "Texas" [ref=e306] [cursor=pointer]
                  - generic [ref=e307]: 51 caucus seats for control · projected margin −1.1 pp
                  - generic [ref=e308]:
                    - text: "Most frequent in simulations:"
                    - button "Ohio" [ref=e309] [cursor=pointer]
                    - text: · 13.1%
              - generic [ref=e310]:
                - generic [ref=e311]: R
                - generic [ref=e312]:
                  - strong [ref=e313]:
                    - button "Texas" [ref=e314] [cursor=pointer]
                  - generic [ref=e315]: 50 caucus seats for control · projected margin +1.1 pp
                  - generic [ref=e316]:
                    - text: "Most frequent in simulations:"
                    - button "Ohio" [ref=e317] [cursor=pointer]
                    - text: · 11.9%
            - generic [ref=e318]:
              - strong [ref=e319]: Ohio · R +0.3 pp
              - generic [ref=e320]: At the D control threshold in 13.1% of simulations.
              - generic [ref=e321]: At the R control threshold in 11.9% of simulations.
            - paragraph [ref=e322]: "A margin is the difference in vote shares: 52% versus 48% is a 4 percentage-point (pp) lead. The frequencies above count how often a race occupies the control-threshold rank across simulated elections, including simulations where that party falls short. They are not the chance that flipping that race alone would change control."
            - group [ref=e323]:
              - generic "How to read the thresholds" [ref=e324] [cursor=pointer]
        - region "Senate race explorer" [ref=e325]:
          - generic [ref=e326]:
            - generic [ref=e327]:
              - generic [ref=e328]:
                - generic [ref=e329]:
                  - heading "A nation of distinct races" [level=2] [ref=e330]
                  - generic [ref=e331]: Select a state to explore its election.
                - generic [ref=e332]: 2026 Senate
              - group "United States Senate races. Use Tab to select a state with an election." [ref=e333]:
                - button "New Mexico, 1 election, Safe" [ref=e334] [cursor=pointer]
                - button "South Dakota, 1 election, Safe" [ref=e335] [cursor=pointer]
                - generic "California, no Senate election" [ref=e336]
                - button "Kentucky, 1 election, Safe" [ref=e337] [cursor=pointer]
                - button "Alabama, 1 election, Safe" [ref=e338] [cursor=pointer]
                - button "Georgia, 1 election, Safe" [ref=e339] [cursor=pointer]
                - button "Arkansas, 1 election, Safe" [ref=e340] [cursor=pointer]
                - generic "Pennsylvania, no Senate election" [ref=e341]
                - generic "Missouri, no Senate election" [ref=e342]
                - button "Colorado, 1 election, Safe" [ref=e343] [cursor=pointer]
                - generic "Utah, no Senate election" [ref=e344]
                - button "Oklahoma, 1 election, Safe" [ref=e345] [cursor=pointer]
                - button "Tennessee, 1 election, Safe" [ref=e346] [cursor=pointer]
                - button "Wyoming, 1 election, Safe" [ref=e347] [cursor=pointer]
                - generic "New York, no Senate election" [ref=e348]
                - generic "Indiana, no Senate election" [ref=e349]
                - button "Kansas, 1 election, Likely" [ref=e350] [cursor=pointer]
                - button "Idaho, 1 election, Likely" [ref=e351] [cursor=pointer]
                - button "Alaska, 1 election, Likely" [ref=e352] [cursor=pointer]
                - generic "Nevada, no Senate election" [ref=e353]
                - button "Illinois, 1 election, Safe" [ref=e354] [cursor=pointer]
                - generic "Vermont, no Senate election" [ref=e355]
                - button "New Jersey, 1 election, Safe" [ref=e356] [cursor=pointer]
                - generic "North Dakota, no Senate election" [ref=e357]
                - button "Iowa, 1 election, Likely" [ref=e358] [cursor=pointer]
                - button "South Carolina, 1 election, Likely" [ref=e359] [cursor=pointer]
                - generic "Arizona, no Senate election" [ref=e360]
                - button "Delaware, 1 election, Safe" [ref=e361] [cursor=pointer]
                - generic "District of Columbia, no Senate election" [ref=e362]
                - generic "Connecticut, no Senate election" [ref=e363]
                - button "New Hampshire, 1 election, Safe" [ref=e364] [cursor=pointer]
                - button "Nebraska, 1 election, Likely" [ref=e365] [cursor=pointer]
                - generic "Washington, no Senate election" [ref=e366]
                - button "Texas, 1 election, Tossup" [ref=e367] [cursor=pointer]
                - button "Ohio, 1 election, Tossup" [pressed] [ref=e368] [cursor=pointer]
                - generic "Wisconsin, no Senate election" [ref=e369]
                - button "Oregon, 1 election, Safe" [ref=e370] [cursor=pointer]
                - button "Mississippi, 1 election, Safe" [ref=e371] [cursor=pointer]
                - button "North Carolina, 1 election, Safe" [ref=e372] [cursor=pointer]
                - button "Virginia, 1 election, Safe" [ref=e373] [cursor=pointer]
                - button "West Virginia, 1 election, Safe" [ref=e374] [cursor=pointer]
                - button "Louisiana, 1 election, Likely" [ref=e375] [cursor=pointer]
                - button "Michigan, 1 election, Likely" [ref=e376] [cursor=pointer]
                - button "Massachusetts, 1 election, Safe" [ref=e377] [cursor=pointer]
                - button "Florida, 1 election, Safe" [ref=e378] [cursor=pointer]
                - button "Montana, 1 election, Safe" [ref=e379] [cursor=pointer]
                - button "Minnesota, 1 election, Likely" [ref=e380] [cursor=pointer]
                - generic "Maryland, no Senate election" [ref=e381]
                - button "Maine, 1 election, Likely" [ref=e382] [cursor=pointer]
                - generic "Hawaii, no Senate election" [ref=e383]
                - button "Rhode Island, 1 election, Safe" [ref=e384] [cursor=pointer]
                - generic: NM
                - generic: SD
                - generic: CA
                - generic: KY
                - generic: AL
                - generic: GA
                - generic: AR
                - generic: PA
                - generic: MO
                - generic: CO
                - generic: UT
                - generic: OK
                - generic: TN
                - generic: WY
                - generic: NY
                - generic: IN
                - generic: KS
                - generic: ID
                - generic: AK
                - generic: NV
                - generic: IL
                - generic: VT
                - generic: ND
                - generic: IA
                - generic: SC
                - generic: AZ
                - generic: NH
                - generic: NE
                - generic: WA
                - generic: TX
                - generic: OH
                - generic: WI
                - generic: OR
                - generic: MS
                - generic: NC
                - generic: VA
                - generic: WV
                - generic: LA
                - generic: MI
                - generic: FL
                - generic: MT
                - generic: MN
                - generic: ME
                - generic: HI
                - generic [ref=e385]: Alaska and Hawaii shown in insets; scale varies.
              - generic [ref=e386]:
                - generic [ref=e387]: Safe D
                - generic [ref=e389]: Likely D
                - generic [ref=e391]: Safe R
                - generic [ref=e393]: Likely R
                - generic [ref=e395]: Safe I
                - generic [ref=e397]: Likely I
                - generic [ref=e399]: Tossup
              - generic [ref=e401]:
                - generic [ref=e402]: "Ratings use the favorite’s chance of winning: Safe above 90%; Likely above 60% through 90%; Tossup 60% or lower (40–60% each in a two-outcome race). Safe does not mean certain."
                - generic [ref=e403]: "D = Democratic · R = Republican · I = Independent. Gray: no election. State area does not represent votes."
            - region [ref=e404]:
              - generic [ref=e406]:
                - heading "Who could control the chamber?" [level=2] [ref=e407]
                - generic [ref=e408]: Independents unaligned · D needs 51; R needs 50
              - generic [ref=e409]:
                - generic [ref=e410]:
                  - generic [ref=e411]: Democratic control
                  - generic [ref=e413]: 43.5%
                - generic [ref=e414]:
                  - generic [ref=e415]: Republican control
                  - generic [ref=e417]: 53.2%
                - generic [ref=e418]:
                  - generic [ref=e419]: Control depends on alliances
                  - generic [ref=e421]: 3.3%
              - paragraph [ref=e426]: King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
              - paragraph [ref=e427]: A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.
              - generic [ref=e428]:
                - generic [ref=e429]:
                  - text: "D caucus: 50.1 expected seats"
                  - generic [ref=e430]: (46–55, modeled 90% range)
                - generic [ref=e431]:
                  - text: "R caucus: 49.6 expected seats"
                  - generic [ref=e432]: (44–54, modeled 90% range)
                - generic [ref=e433]:
                  - text: "Unaligned: 0.3 expected seats"
                  - generic [ref=e434]: (0–2, modeled 90% range)
              - paragraph [ref=e435]: Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.
            - region [ref=e436]:
              - heading "What has been tested?" [level=2] [ref=e437]
              - paragraph [ref=e438]: The 2026 election has not happened yet. Select 2018 or 2022 to see how historical reconstructions compare with actual results. Those views use a different model of possible winners, so their scores do not measure the accuracy of this 2026 two-contender forecast.
              - group [ref=e439]:
                - generic "Scope and limitations" [ref=e440] [cursor=pointer]
            - region [ref=e441]:
              - generic [ref=e442]:
                - heading "The races" [level=2] [ref=e443]
                - generic [ref=e444]: 10 of 35 elections
              - paragraph [ref=e445]: The favorite is the candidate or group with the highest modeled chance of winning. “Most competitive” puts the least certain favorites first.
              - generic [ref=e446]:
                - generic [ref=e447]:
                  - generic [aria-hidden] [ref=e448]: ⌕
                  - textbox "Search states or candidates" [ref=e449]:
                    - /placeholder: Find a state or candidate
                - combobox "Sort races" [ref=e450]:
                  - option "Most competitive" [selected]
                  - option "State A–Z"
              - table [ref=e451]:
                - rowgroup [ref=e452]:
                  - row [ref=e453]:
                    - columnheader "State / election" [ref=e454]
                    - columnheader "Favorite" [ref=e455]
                    - columnheader "Chance to win" [ref=e456]
                    - columnheader "Outlook" [ref=e457]
                - rowgroup [ref=e458]:
                  - row [ref=e459]:
                    - cell [ref=e460]:
                      - button "Ohio Special election" [pressed] [ref=e461] [cursor=pointer]:
                        - text: Ohio
                        - generic [ref=e462]: Special election
                    - cell "Jon Husted Republican" [ref=e463]:
                      - text: Jon Husted
                      - generic [ref=e464]: Republican
                    - cell "52.6%" [ref=e465]
                    - cell "Tossup" [ref=e469]
                  - row [ref=e470]:
                    - cell [ref=e471]:
                      - button "Texas Regular election" [ref=e472] [cursor=pointer]:
                        - text: Texas
                        - generic [ref=e473]: Regular election
                    - cell "KEN PAXTON Republican" [ref=e474]:
                      - text: KEN PAXTON
                      - generic [ref=e475]: Republican
                    - cell "58.6%" [ref=e476]
                    - cell "Tossup" [ref=e480]
                  - row [ref=e481]:
                    - cell [ref=e482]:
                      - button "Maine Regular election" [ref=e483] [cursor=pointer]:
                        - text: Maine
                        - generic [ref=e484]: Regular election
                    - cell "Jackson, Troy D. Democratic" [ref=e485]:
                      - text: Jackson, Troy D.
                      - generic [ref=e486]: Democratic
                    - cell "63.0%" [ref=e487]
                    - cell "Likely" [ref=e491]
                  - row [ref=e492]:
                    - cell [ref=e493]:
                      - button "Alaska Regular election" [ref=e494] [cursor=pointer]:
                        - text: Alaska
                        - generic [ref=e495]: Regular election
                    - cell "Sullivan, Dan S. Republican" [ref=e496]:
                      - text: Sullivan, Dan S.
                      - generic [ref=e497]: Republican
                    - cell "64.4%" [ref=e498]
                    - cell "Likely" [ref=e502]
                  - row [ref=e503]:
                    - cell [ref=e504]:
                      - button "Iowa Regular election" [ref=e505] [cursor=pointer]:
                        - text: Iowa
                        - generic [ref=e506]: Regular election
                    - cell "Ashley Hinson Republican" [ref=e507]:
                      - text: Ashley Hinson
                      - generic [ref=e508]: Republican
                    - cell "68.6%" [ref=e509]
                    - cell "Likely" [ref=e513]
                  - row [ref=e514]:
                    - cell [ref=e515]:
                      - button "Michigan Regular election" [ref=e516] [cursor=pointer]:
                        - text: Michigan
                        - generic [ref=e517]: Regular election
                    - cell "Abdul El-Sayed Democratic" [ref=e518]:
                      - text: Abdul El-Sayed
                      - generic [ref=e519]: Democratic
                    - cell "68.8%" [ref=e520]
                    - cell "Likely" [ref=e524]
                  - row [ref=e525]:
                    - cell [ref=e526]:
                      - button "Kansas Regular election" [ref=e527] [cursor=pointer]:
                        - text: Kansas
                        - generic [ref=e528]: Regular election
                    - cell "Roger Marshall Republican" [ref=e529]:
                      - text: Roger Marshall
                      - generic [ref=e530]: Republican
                    - cell "74.5%" [ref=e531]
                    - cell "Likely" [ref=e535]
                  - row [ref=e536]:
                    - cell [ref=e537]:
                      - button "Minnesota Regular election" [ref=e538] [cursor=pointer]:
                        - text: Minnesota
                        - generic [ref=e539]: Regular election
                    - cell "Peggy Flanagan Democratic" [ref=e540]:
                      - text: Peggy Flanagan
                      - generic [ref=e541]: Democratic
                    - cell "80.3%" [ref=e542]
                    - cell "Likely" [ref=e546]
                  - row [ref=e547]:
                    - cell [ref=e548]:
                      - button "South Carolina Regular election" [ref=e549] [cursor=pointer]:
                        - text: South Carolina
                        - generic [ref=e550]: Regular election
                    - cell "Darline Graham Republican" [ref=e551]:
                      - text: Darline Graham
                      - generic [ref=e552]: Republican
                    - cell "82.7%" [ref=e553]
                    - cell "Likely" [ref=e557]
                  - row [ref=e558]:
                    - cell [ref=e559]:
                      - button "Nebraska Regular election" [ref=e560] [cursor=pointer]:
                        - text: Nebraska
                        - generic [ref=e561]: Regular election
                    - cell "Pete Ricketts Republican" [ref=e562]:
                      - text: Pete Ricketts
                      - generic [ref=e563]: Republican
                    - cell "84.4%" [ref=e564]
                    - cell "Likely" [ref=e568]
              - button "See all 35 races" [ref=e569] [cursor=pointer]:
                - text: See all 35 races
                - generic [aria-hidden] [ref=e570]: ⌄
          - complementary "Selected race detail" [ref=e571]:
            - generic [ref=e572]:
              - generic [ref=e573]: Race in focus
              - generic [ref=e574]: Tossup
            - heading "Ohio" [level=2] [ref=e575]
            - generic [ref=e576]: U.S. Senate · Special election
            - separator [ref=e577]
            - heading "Chance of winning" [level=3] [ref=e578]
            - paragraph [ref=e579]: A 70% chance means winning about 70 out of 100 elections simulated by this model. It does not mean winning 70% of the votes.
            - generic [ref=e580]:
              - generic [ref=e582]:
                - generic [ref=e583]:
                  - generic [ref=e584]: Sherrod Brown
                  - generic [ref=e585]: Democratic
                - strong [ref=e586]: 47.4%
              - generic [ref=e590]:
                - generic [ref=e591]:
                  - generic [ref=e592]: Jon Husted
                  - generic [ref=e593]: Republican · Currently holds this seat
                - strong [ref=e594]: 52.6%
            - separator [ref=e597]
            - heading "Vote share among these two contenders" [level=3] [ref=e598]
            - generic [ref=e599]:
              - generic [ref=e600]:
                - generic [ref=e601]:
                  - generic [ref=e602]: Brown
                  - strong [ref=e603]: 49.8%
                - 'generic "Sherrod Brown: mean 49.8%, Modeled 90% range 45.6% to 54.2%" [ref=e604]'
                - paragraph [ref=e607]: "Modeled 90% range: 45.6%–54.2%Share of votes for the two named principal contenders"
              - generic [ref=e608]:
                - generic [ref=e609]:
                  - generic [ref=e610]: Husted
                  - strong [ref=e611]: 50.2%
                - 'generic "Jon Husted: mean 50.2%, Modeled 90% range 45.8% to 54.4%" [ref=e612]'
                - paragraph [ref=e615]: "Modeled 90% range: 45.8%–54.4%Share of votes for the two named principal contenders"
            - paragraph [ref=e616]: The shares divide votes between these two contenders only; votes for other candidates are excluded. The range runs from the 5th to the 95th percentile of simulated shares—the middle 90%. It is a model estimate, not a poll’s margin of error or a guarantee.
            - paragraph [ref=e617]: Probabilities of 0% or 100% reflect the saved simulation, not certainty.
            - separator [ref=e618]
            - generic [ref=e619]:
              - generic [ref=e620]:
                - text: Selected model setup
                - strong [ref=e621]: Standard polling
              - generic [ref=e622]:
                - text: Poll records included
                - strong [ref=e623]: "11"
            - paragraph [ref=e624]:
              - text: This counts poll records used for this race, not people interviewed or every poll available.
              - link "Why some polls are included—and others are not ↓" [ref=e625] [cursor=pointer]:
                - /url: "#poll-selection"
            - group [ref=e626]:
              - generic "Race assumptions" [ref=e627] [cursor=pointer]
            - separator [ref=e628]
            - heading "Candidate and result sources" [level=3] [ref=e629]
            - generic [ref=e630]:
              - link "Sherrod Brown ↗" [ref=e632] [cursor=pointer]:
                - /url: https://www.boe.ohio.gov/abmt/oh12/20261103/X001___2_CITY_1.html
              - link "Jon Husted ↗" [ref=e634] [cursor=pointer]:
                - /url: https://www.boe.ohio.gov/abmt/oh12/20261103/X001___2_CITY_1.html
            - paragraph [ref=e635]: "Forecast date: October 26, 2026Inputs dated through: October 26, 2026Saved research forecast; new polls are not added automatically."
        - region [ref=e636]:
          - generic [ref=e637]:
            - generic [ref=e638]: From evidence to possible elections
            - heading "How this forecast works." [level=2] [ref=e639]
            - paragraph [ref=e640]: A model is a set of mathematical assumptions connecting evidence to possible results. Your choices change those assumptions and, when available, load the matching saved forecast.
            - link "Download this forecast’s data ↓" [ref=e641] [cursor=pointer]:
              - /url: /data/explorer/current-polling.json
            - paragraph [ref=e642]: "The download is a JSON file: structured data for analysis, with source identifiers and limitations."
          - generic [ref=e643]:
            - generic [ref=e644]:
              - heading "1. Start with a baseline" [level=3] [ref=e645]
              - paragraph [ref=e646]: Electoral history supplies a starting estimate, often called “fundamentals.” The builder also offers a historical winner reference model. A component is one part of the model; the chosen combination is your model setup or “stack.”
            - generic [ref=e647]:
              - heading "2. Add the selected evidence" [level=3] [ref=e648]
              - paragraph [ref=e649]:
                - text: This setup adds the polling evidence saved for this election. Polls measure responses at a particular time. The model allows for measurement errors and changes before election day.
                - link "Read the poll selection rules below." [ref=e650] [cursor=pointer]:
                  - /url: "#poll-selection"
            - generic [ref=e651]:
              - heading "3. Simulate elections together" [level=3] [ref=e652]
              - paragraph [ref=e653]: A simulation is one possible set of election results generated by the model. Shared uncertainty allows many races to move together—for example, if polls miss in the same direction. Win chances and chamber control summarize these simulated elections, rather than assuming each race is unrelated.
            - generic [ref=e654]:
              - heading "4. Explore what changes" [level=3] [ref=e655]
              - paragraph [ref=e656]: Changing a choice loads a saved result if one exists. Some combinations require calculating several model parts together again (a “joint refit”); the website explains when that work is needed. A sensitivity test changes assumptions without adding evidence. “Runs successfully,” “experimental,” and “tested against election results” describe different things.
            - generic [ref=e657]:
              - heading "What these numbers cover" [level=3] [ref=e658]
              - paragraph [ref=e659]: The 2026 preview models only the two named contenders in each race. Its vote shares divide votes between those two; other candidates’ shares and chances of winning are outside this model.
            - generic [ref=e660]:
              - heading "Make fair comparisons" [level=3] [ref=e661]
              - paragraph [ref=e662]: Compare the same year, possible winners and assumptions about independent senators. A change from 40% to 45% is 5 percentage points (“pp”), not a 5% relative increase. Historical scores describe those elections and do not establish the accuracy of the current forecast.
        - region [ref=e663]:
          - generic [ref=e664]: The evidence behind this view
          - heading "Why some polls are included—and others are not" [level=2] [ref=e665]
          - paragraph [ref=e666]:
            - text: This saved forecast uses
            - strong [ref=e667]: 131 poll records across 25 of 35 races
            - text: ", from sources collected by October 26, 2026. A record is one selected survey question for one race. It is not a count of people interviewed, and this collection does not include every published poll."
          - generic [ref=e668]:
            - generic [ref=e669]:
              - heading "Match the contest and the date" [level=3] [ref=e670]
              - paragraph [ref=e671]: The race and named contenders must match the forecast. Sources must document usable percentages and show that the evidence was available by the saved cutoff. Fieldwork must end within 100 days of election day. An availability check does not establish the exact day a poll was published.
            - generic [ref=e672]:
              - heading "Avoid counting the same evidence twice" [level=3] [ref=e673]
              - paragraph [ref=e674]: Only one question is chosen from a survey, so alternative ballot questions do not become extra polls. For each race, the latest eligible sample in each polling-provider grouping is retained. Where the actual survey contractor is unknown, the named survey organization supplies that grouping; this is an approximation, not a verified contractor identity.
            - generic [ref=e675]:
              - heading "Keep source gaps visible" [level=3] [ref=e676]
              - paragraph [ref=e677]: Missing or uncertain source documents, dates or suitable questions can keep a record out. An absent poll may simply be outside the collected sources. Its absence is not, by itself, a judgment that the pollster is unreliable.
          - group [ref=e678]:
            - generic "How sample sizes, voter types and missing polls are handled" [ref=e679] [cursor=pointer]
          - group [ref=e680]:
            - generic "Check included records and recorded exclusions by state" [ref=e681] [cursor=pointer]
            - option "All states"
            - option "Alabama"
            - option "Alaska"
            - option "Arkansas"
            - option "Colorado"
            - option "Delaware"
            - option "Florida"
            - option "Georgia"
            - option "Idaho"
            - option "Illinois"
            - option "Iowa"
            - option "Kansas"
            - option "Kentucky"
            - option "Louisiana"
            - option "Maine"
            - option "Massachusetts"
            - option "Michigan"
            - option "Minnesota"
            - option "Mississippi"
            - option "Montana"
            - option "Nebraska"
            - option "New Hampshire"
            - option "New Jersey"
            - option "New Mexico"
            - option "North Carolina"
            - option "Ohio" [selected]
            - option "Oklahoma"
            - option "Oregon"
            - option "Rhode Island"
            - option "South Carolina"
            - option "South Dakota"
            - option "Tennessee"
            - option "Texas"
            - option "Virginia"
            - option "West Virginia"
            - option "Wyoming"
        - generic [ref=e682]:
          - generic [ref=e683]: ballotmancy
          - generic [ref=e684]: Independent research. Visible assumptions.2026 Senate research preview · October 26, 2026
  - alert [ref=e685]
```

# Test source

```ts
  112 |   await expect(detail.locator(`[aria-label="${shareLabel(rickettsFundamentals)}"]`)).toBeVisible();
  113 |   await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_R");
  114 |   await expect(chamber.getByText(percent(fundamentalsAlignedR.control.republicanMajorityProbability), { exact: true })).toBeVisible();
  115 |   await page.getByLabel("Who joins each party?").selectOption("campaign_pledge_unaligned");
  116 |   const fundamentalsUnaligned = fundamentals.scenarios.find((item: { id: string }) => item.id === "campaign_pledge_unaligned");
  117 |   await expect(chamber.getByText(percent(fundamentalsUnaligned.control.unresolvedOrganizingProbability), { exact: true })).toBeVisible();
  118 |   expect(errors).toEqual([]);
  119 |   expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  120 | });
  121 | 
  122 | test("race search, special elections, and map keyboard selection work together", async ({ page }) => {
  123 |   await page.goto("/");
  124 |   await expandControls(page);
  125 |   const detail = page.getByRole("complementary", { name: "Selected race detail" });
  126 |   const search = page.getByRole("textbox", { name: "Search states or candidates" });
  127 |   await search.fill("special");
  128 |   await expect(page.locator(".race-table tbody tr")).toHaveCount(2);
  129 |   await page.getByRole("button", { name: "Florida Special election", exact: true }).click();
  130 |   await expect(detail.getByRole("heading", { name: "Florida", exact: true })).toBeVisible();
  131 |   await expect(detail.getByText("U.S. Senate · Special election", { exact: true })).toBeVisible();
  132 |   await search.fill("no such contender");
  133 |   await expect(page.getByText("No matching races. Try another state or candidate.", { exact: true })).toBeVisible();
  134 |   await search.fill("");
  135 |   await page.getByLabel("Sort races").selectOption("state");
  136 |   await expect(page.locator(".race-table tbody tr").first()).toContainText("Alabama");
  137 |   const alaska = page.getByRole("button", { name: /^Alaska, 1 election,/ });
  138 |   await alaska.focus();
  139 |   await page.keyboard.press("Enter");
  140 |   await expect(detail.getByRole("heading", { name: "Alaska", exact: true })).toBeVisible();
  141 |   await expect(alaska).toHaveAttribute("aria-pressed", "true");
  142 | });
  143 | 
  144 | test("the race list remains usable when map geography fails", async ({ page }) => {
  145 |   await page.route("**/states-us.geojson", (route) => route.abort());
  146 |   await page.goto("/");
  147 |   await expandControls(page);
  148 |   await expect(page.getByText("Map unavailable. Explore every race in the list below.", { exact: true })).toBeVisible();
  149 |   await page.getByRole("button", { name: "Texas Regular election", exact: true }).click();
  150 |   await expect(page.getByRole("complementary").getByRole("heading", { name: "Texas", exact: true })).toBeVisible();
  151 | });
  152 | 
  153 | test("joint-draw tipping frequencies retain the correct caucus thresholds and denominator", async ({ request }) => {
  154 |   const response = await request.get("/data/tipping-points.json");
  155 |   expect(response.ok()).toBeTruthy();
  156 |   const tipping = await response.json() as TippingSnapshot;
  157 |   const forecast = await (await request.get("/data/forecast.json")).json() as SiteForecastSnapshot;
  158 |   expect(tipping.method).toBe("caucus_rank_threshold_v1");
  159 |   expect(tipping.sources.pollingForecastSha256).toBe(forecast.sources.pollingSha256);
  160 |   expect(tipping.sources.fundamentalsForecastSha256).toBe(forecast.sources.fundamentalsSha256);
  161 |   const texas = "2026:TX:senate:regular";
  162 |   const independentIds = ["ID", "MT", "NE", "SD"].map((state) => `2026:${state}:senate:regular`);
  163 |   for (const modelKey of ["polling", "fundamentals"] as const) {
  164 |     const model = tipping.models[modelKey];
  165 |     expect(model.simulationCount).toBe(modelKey === "polling" ? 40000 : 100000);
  166 |     const expectedOrder = [...forecast.models[modelKey].races]
  167 |       .sort((a, b) => b.firstShareMean - a.firstShareMean || a.id.localeCompare(b.id))
  168 |       .map((race) => race.id);
  169 |     expect(model.raceOrder).toEqual(expectedOrder);
  170 |     for (const scenario of Object.values(model.scenarios)) {
  171 |       expect(scenario.D.thresholdSeats).toBe(51);
  172 |       expect(scenario.R.thresholdSeats).toBe(50);
  173 |       expect(scenario.D.retainedSeats).toBe(34);
  174 |       expect(scenario.R.retainedSeats).toBe(31);
  175 |       for (const caucus of [scenario.D, scenario.R]) {
  176 |         expect(caucus.projectedTippingRaceId).toBe(texas);
  177 |         expect(Object.keys(caucus.frequencyByRace).sort()).toEqual([...expectedOrder].sort());
  178 |         expect(Object.values(caucus.frequencyByRace).reduce((sum, p) => sum + p, 0)).toBeCloseTo(1, 10);
  179 |       }
  180 |     }
  181 |     const unaligned = model.scenarios.campaign_pledge_unaligned;
  182 |     expect(unaligned.D.requiredContestedWins).toBe(17);
  183 |     expect(unaligned.R.requiredContestedWins).toBe(19);
  184 |     expect(unaligned.D.eligibleRaceIds).toHaveLength(31);
  185 |     expect(unaligned.R.eligibleRaceIds).toHaveLength(35);
  186 |     expect(unaligned.D.frequencyByRace).not.toEqual(unaligned.R.frequencyByRace);
  187 |     for (const id of independentIds) expect(unaligned.D.frequencyByRace[id]).toBe(0);
  188 |     const alignedD = model.scenarios.independent_opposition_caucuses_D;
  189 |     expect(alignedD.D.frequencyByRace).toEqual(alignedD.R.frequencyByRace);
  190 |     expect(alignedD.D.frequencyByRace).toEqual(unaligned.R.frequencyByRace);
  191 |     const alignedR = model.scenarios.independent_opposition_caucuses_R;
  192 |     expect(alignedR.R.guaranteedRaceIds.sort()).toEqual([...independentIds].sort());
  193 |     expect(alignedR.R.requiredContestedWins).toBe(15);
  194 |     expect(alignedR.D.frequencyByRace).toEqual(alignedR.R.frequencyByRace);
  195 |     expect(alignedR.D.frequencyByRace).toEqual(unaligned.D.frequencyByRace);
  196 |     for (const id of independentIds) expect(alignedR.R.frequencyByRace[id]).toBe(0);
  197 |   }
  198 |   // Independently computed directly from the immutable 40,000 / 100,000 saved draw rows.
  199 |   expect(tipping.models.polling.scenarios.campaign_pledge_unaligned.D.frequencyByRace[texas]).toBe(0.128225);
  200 |   expect(tipping.models.polling.scenarios.campaign_pledge_unaligned.R.frequencyByRace[texas]).toBe(0.102975);
  201 |   expect(tipping.models.fundamentals.scenarios.campaign_pledge_unaligned.D.frequencyByRace[texas]).toBe(0.11148);
  202 |   expect(tipping.models.fundamentals.scenarios.campaign_pledge_unaligned.R.frequencyByRace[texas]).toBe(0.09966);
  203 | });
  204 | 
  205 | test("ribbon displays all margin-ordered races, coincident markers, and scenario-specific tipping frequencies", async ({ page, request }) => {
  206 |   const tipping = await (await request.get("/data/tipping-points.json")).json() as TippingSnapshot;
  207 |   await page.goto("/");
  208 |   await expandControls(page);
  209 |   const ribbon = page.getByRole("region", { name: "The path to a majority", exact: true });
  210 |   const tiles = ribbon.locator("button[data-race-id]");
  211 |   await expect(tiles).toHaveCount(35);
> 212 |   expect(await tiles.evaluateAll((elements) => elements.map((element) => (element as HTMLElement).dataset.raceId))).toEqual(tipping.models.polling.raceOrder);
      |                                                                                                                     ^ Error: expect(received).toEqual(expected) // deep equality
  213 |   await expect(ribbon.locator(".retained-endcap")).toHaveCount(2);
  214 |   await expect(ribbon.locator(".retained-endcap").first()).toContainText("34");
  215 |   await expect(ribbon.locator(".retained-endcap").last()).toContainText("31");
  216 |   await expect(ribbon.getByText("D51 / R50 · TX", { exact: true })).toBeVisible();
  217 |   const texas = ribbon.locator('[data-race-id="2026:TX:senate:regular"]');
  218 |   await expect(texas).toHaveAttribute("aria-label", /projected D control threshold, projected R control threshold/);
  219 |   await texas.click();
  220 |   const selectedFrequencies = ribbon.locator(".selected-threshold");
  221 |   await expect(selectedFrequencies).toContainText("At the D control threshold in 12.8% of simulations.");
  222 |   await expect(selectedFrequencies).toContainText("At the R control threshold in 10.3% of simulations.");
  223 | 
  224 |   await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_D");
  225 |   await expect(selectedFrequencies).toContainText("At the D control threshold in 10.3% of simulations.");
  226 |   const nebraska = ribbon.locator('[data-race-id="2026:NE:senate:regular"]');
  227 |   await nebraska.focus();
  228 |   await page.keyboard.press("Enter");
  229 |   await expect(nebraska).toHaveAttribute("aria-pressed", "true");
  230 |   await expect(page.getByRole("complementary").getByText("Independent", { exact: true })).toBeVisible();
  231 |   await expect(selectedFrequencies).toContainText("At the D control threshold in 3.3% of simulations.");
  232 |   await expect(selectedFrequencies).toContainText("At the R control threshold in 3.3% of simulations.");
  233 |   await page.getByLabel("Who joins each party?").selectOption("independent_opposition_caucuses_R");
  234 |   await expect(nebraska).toHaveAttribute("aria-label", /R caucus regardless of winner/);
  235 |   await expect(selectedFrequencies).toContainText("At the D control threshold in 0.0% of simulations.");
  236 |   await expect(selectedFrequencies).toContainText("At the R control threshold in 0.0% of simulations.");
  237 |   await expect(page.getByRole("complementary").getByText("Independent", { exact: true })).toBeVisible();
  238 |   await expect(ribbon.locator(".ribbon-guarantee")).toContainText("4 contests count toward the R caucus");
  239 | 
  240 |   await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).uncheck();
  241 |   await expect(page.locator(".stack-builder")).toHaveAttribute("aria-busy", "false");
  242 |   expect(await tiles.evaluateAll((elements) => elements.map((element) => (element as HTMLElement).dataset.raceId))).toEqual(tipping.models.fundamentals.raceOrder);
  243 |   await texas.click();
  244 |   await expect(selectedFrequencies).toContainText("At the D control threshold in 11.1% of simulations.");
  245 |   await expect(selectedFrequencies).toContainText("At the R control threshold in 11.1% of simulations.");
  246 |   await ribbon.getByText("How to read the thresholds", { exact: true }).click();
  247 |   await expect(ribbon.getByText(/Simulations where the party group falls short still count/)).toBeVisible();
  248 |   expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  249 |   expect(await ribbon.evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector(".map-card")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBeTruthy();
  250 |   expect(await page.locator(".map-card").evaluate((element) => Boolean(element.compareDocumentPosition(document.querySelector(".control-card")!) & Node.DOCUMENT_POSITION_FOLLOWING))).toBeTruthy();
  251 | });
  252 | 
  253 | // Multi-contest historical states and atomic bundle failures are covered in explorer.spec.ts.
  254 | 
```