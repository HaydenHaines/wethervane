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
        - generic [ref=e8]: Research preview
    - generic [ref=e10]:
      - tablist "Chamber" [ref=e11]:
        - tab "Senate" [selected] [ref=e12] [cursor=pointer]
        - tab "House" [ref=e13] [cursor=pointer]
      - generic [ref=e14]: Two chambers. Your perspective.
    - tabpanel "Senate" [ref=e15]:
      - main [ref=e16]:
        - generic [ref=e18]:
          - generic [ref=e19]: 2026 United States Senate · 35 elections
          - heading "The Senate, in perspective." [level=1] [ref=e20]: The Senate,in perspective.
          - paragraph [ref=e21]: Choose the model’s ingredients. Explore the races and chamber together.A closer look at the races that could shape the chamber, with visible uncertainty.
          - group [ref=e22]:
            - generic "Input and publication times" [ref=e23] [cursor=pointer]
        - group [ref=e24]:
          - generic "Build your perspective Standard polling Collapse controls" [active] [ref=e25] [cursor=pointer]:
            - generic [ref=e26]:
              - generic [ref=e27]: Build your perspective
              - generic [ref=e28]: Standard polling
            - generic [ref=e29]:
              - generic [ref=e30]: Collapse controls
              - generic [aria-hidden] [ref=e31]: ⌄
          - generic [ref=e32]:
            - generic [ref=e33]:
              - generic [ref=e34]:
                - generic [ref=e35]: Customize this view
                - generic [ref=e36]: A stack is your combination of model choices. Select saved results or inspect experimental alternatives; experimental choices may need a new calculation before they can be shown.
              - generic [ref=e37]:
                - generic [ref=e38] [cursor=pointer]:
                  - checkbox "Compare default" [ref=e39]
                  - text: Compare default
                - button "Reset stack" [ref=e40] [cursor=pointer]
                - button "Share view ↗" [ref=e41] [cursor=pointer]
            - generic [ref=e42]:
              - generic [ref=e43]:
                - generic [ref=e44]: Election year
                - group "Election year" [ref=e45]:
                  - button "2026" [pressed] [ref=e46] [cursor=pointer]
                  - button "2022" [ref=e47] [cursor=pointer]
                  - button "2018" [ref=e48] [cursor=pointer]
              - generic [ref=e49]:
                - generic [ref=e50]: Forecast date
                - combobox "Forecast date" [ref=e51]:
                  - option "October 20, 2026"
                  - option "October 26, 2026" [selected]
                - paragraph [ref=e52]: Choose an available forecast date. Each date has its own saved component combinations; changing a component can switch to its saved edition.
            - generic [ref=e53]:
              - generic [ref=e54]:
                - strong [ref=e55]: Explore individual model components
                - paragraph [ref=e56]: A 2018 study separates more of the model’s ingredients. It covers 30 Democratic–Republican races and has its own race scorecard.
              - button "Explore the 2018 component study" [ref=e57] [cursor=pointer]
            - generic [ref=e58]:
              - generic [ref=e59]: Preserved presets · ready-made combinations
              - group "Preserved forecast presets" [ref=e60]:
                - button "2026 Basic electoral history v5" [ref=e61] [cursor=pointer]:
                  - generic [ref=e62]: "2026"
                  - text: Basic electoral history v5
                - button "2026 Best guess with October 26 polling" [pressed] [ref=e63] [cursor=pointer]:
                  - generic [ref=e64]: "2026"
                  - text: Best guess with October 26 polling
            - status [ref=e65]:
              - strong [ref=e66]: Ready
              - paragraph [ref=e67]:
                - generic [ref=e68]: "Displaying:"
                - strong [ref=e69]: Standard polling
                - text: · Research preview · October 26, 2026 (2026-10-26).
            - region [ref=e70]:
              - generic [ref=e71]:
                - heading "The full model, slice by slice" [level=3] [ref=e72]
                - paragraph [ref=e73]: Choose the ingredients for your requested setup. Results change when a matching saved forecast loads. Unfinished slices stay visible as the model develops.
              - generic [ref=e74]:
                - region [ref=e75]:
                  - generic [ref=e76]:
                    - generic [ref=e77]:
                      - checkbox "Include population & geography" [disabled] [ref=e78]
                      - heading "Population & geography" [level=4] [ref=e79]
                      - generic [ref=e80]: Not connected yet
                    - paragraph [ref=e81]: Defines who is represented, where, and for which dates.
                  - paragraph [ref=e82]: A population model is not connected yet. These forecasts use a fixed list of contests.
                - region [ref=e83]:
                  - generic [ref=e84]:
                    - generic [ref=e85]:
                      - checkbox "Include community composition" [disabled] [ref=e86]
                      - heading "Community composition" [level=4] [ref=e87]
                      - generic [ref=e88]: Not connected yet
                    - paragraph [ref=e89]: Models measured groups and uncertainty about how they overlap.
                  - paragraph [ref=e90]: Community composition is not estimated in these forecasts.
                - region [ref=e91]:
                  - generic [ref=e92]:
                    - generic [ref=e93]:
                      - generic [aria-hidden] [ref=e94]: ✓
                      - heading "Electoral history" [level=4] [ref=e95]
                      - generic [ref=e96]: Always included
                    - paragraph [ref=e97]: Sets a starting estimate from earlier elections and candidate context.
                  - generic [ref=e99]:
                    - generic [ref=e100]: Base model
                    - combobox "Base model" [ref=e101]:
                      - option "Electoral-history starting estimate" [selected]
                      - option "Recent calendar history"
                      - option "Refreshed history"
                      - option "Learned presidential slope"
                      - option "Relative lean with presidential context"
                      - option "Relative state and national lean"
                    - paragraph [ref=e102]: Starts from electoral history, combining information into a race-level estimate before adding any selected polls. This is the saved fundamentals model.
                - region [ref=e103]:
                  - generic [ref=e104]:
                    - generic [ref=e105]:
                      - generic [aria-hidden] [ref=e106]: ✓
                      - heading "National environment" [level=4] [ref=e107]
                      - generic [ref=e108]: Always included
                    - paragraph [ref=e109]: Sets uncertainty and how strongly races can move together.
                  - generic [ref=e111]:
                    - generic [ref=e112]: Uncertainty
                    - combobox "Uncertainty" [ref=e113]:
                      - option "Standard uncertainty" [selected]
                      - option "Polling-error sizes × 2/3"
                      - option "Polling-error sizes × 1.5"
                      - option "Shared election-wide error size × 0.5"
                      - option "Shared election-wide error size × 2"
                    - paragraph [ref=e114]: Uses the saved default assumptions about how much results can vary and how races can move together.
                - region [ref=e115]:
                  - generic [ref=e116]:
                    - generic [ref=e117]:
                      - generic [aria-hidden] [ref=e118]: ✓
                      - heading "Polls & surveys" [level=4] [ref=e119]
                      - generic [ref=e120]: Included
                    - paragraph [ref=e121]: Adds polling evidence while allowing for survey error.
                  - generic [ref=e123]:
                    - generic [ref=e124] [cursor=pointer]:
                      - checkbox "Include polling evidence" [checked] [ref=e125]
                      - text: Include polling evidence
                    - paragraph [ref=e126]: Standard polling
                    - paragraph [ref=e127]: "Adds polls using a fixed bell-shaped (Gaussian) model for measurement error: polls are evidence with uncertainty, not exact election results. Only saved combinations can be displayed."
                - region [ref=e128]:
                  - generic [ref=e129]:
                    - generic [ref=e130]:
                      - checkbox "Include turnout & participation" [disabled] [ref=e131]
                      - heading "Turnout & participation" [level=4] [ref=e132]
                      - generic [ref=e133]: Not connected yet
                    - paragraph [ref=e134]: Models who votes and how turnout varies.
                  - paragraph [ref=e135]: A turnout model is not connected yet. These forecasts do not predict voter counts.
                - region [ref=e136]:
                  - generic [ref=e137]:
                    - generic [ref=e138]:
                      - generic [aria-hidden] [ref=e139]: ✓
                      - heading "Candidate choice" [level=4] [ref=e140]
                      - generic [ref=e141]: Built into this model
                    - paragraph [ref=e142]: Models choices among candidates and ballot options.
                  - paragraph [ref=e143]: The candidate options are fixed in each saved forecast. There is no separate candidate-choice control.
                - region [ref=e144]:
                  - generic [ref=e145]:
                    - generic [ref=e146]:
                      - generic [aria-hidden] [ref=e147]: ✓
                      - heading "Vote counts & election rules" [level=4] [ref=e148]
                      - generic [ref=e149]: Built into this model
                    - paragraph [ref=e150]: Turns voting choices into counts, winners and chamber outcomes.
                  - paragraph [ref=e151]: Chamber calculations and tie rules are fixed in each saved Senate result. Total vote counts are not predicted, and there is no separate control to change these rules.
              - paragraph [ref=e152]: “Always included” means this part is required by these saved models. “Built into this model” means it is present but has no separate control. Disabled choices cannot change this result. These choices belong to one model with shared assumptions; they are not percentages to blend.
            - generic [ref=e153]:
              - generic [ref=e154]:
                - generic [ref=e155]: Comparison baseline
                - combobox "Comparison baseline" [ref=e156]:
                  - option "2026 default" [selected]
                  - 'option "Polling error: shared election-wide error size × 2"'
                  - 'option "Polling error: shared election-wide error size × 0.5"'
                  - 'option "Polling error: all error sizes × 1.5"'
                  - 'option "Polling error: all error sizes × 2/3"'
                - paragraph [ref=e157]: A baseline is the reference result you compare against. Choices must predict the same election and type of outcome on the same date. The historical scorecard uses this reference; the checkbox adds comparison details below.
              - group [ref=e158]:
                - generic "Saved views (0) · This browser" [ref=e159] [cursor=pointer]:
                  - text: Saved views
                  - generic [ref=e160]: (0) · This browser
            - group [ref=e161]:
              - generic "Configuration, evidence & data dates" [ref=e162] [cursor=pointer]
        - generic [ref=e163]:
          - generic [ref=e164]:
            - generic [ref=e165]: Standard polling · 40,000 saved simulations
            - generic [ref=e166]:
              - generic [ref=e167]: Who joins each party?
              - combobox "Who joins each party?" [ref=e168]:
                - option "Independents unaligned" [selected]
                - option "Independents work with Democrats"
                - option "Independents work with Republicans"
          - paragraph [ref=e169]: Each simulation is one possible set of election results. A caucus is a party’s group in the Senate, including independents who work with it. Choose those alliances above; candidate party labels stay the same.
          - region [ref=e170]:
            - generic [ref=e171]:
              - generic [ref=e172]:
                - heading "The path to a majority" [level=2] [ref=e173]
                - paragraph [ref=e174]: 35 elections, ordered by the projected vote-share lead between the two modeled contenders.
              - generic [ref=e175]:
                - generic [ref=e176]: D
                - generic [ref=e177]: R
                - text: Projected control thresholds
            - generic [aria-hidden] [ref=e178]:
              - generic [ref=e179]: Democratic / independent advantage
              - generic [ref=e180]: Republican advantage
            - region "Senate race ribbon, ordered by projected margin. Scroll horizontally for all 35 elections." [ref=e181]:
              - generic [ref=e182]:
                - generic [ref=e183]:
                  - strong [ref=e184]: "34"
                  - generic [ref=e185]: D caucusnot up this year
                - button "Rhode Island, regular election, Safe, projected margin D +31.4 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e186] [cursor=pointer]:
                  - generic [ref=e187]: RI
                  - generic [ref=e188]: ’26
                - button "Massachusetts, regular election, Safe, projected margin D +30.8 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e189] [cursor=pointer]:
                  - generic [ref=e190]: MA
                  - generic [ref=e191]: ’26
                - button "Oregon, regular election, Safe, projected margin D +28.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e192] [cursor=pointer]:
                  - generic [ref=e193]: OR
                  - generic [ref=e194]: ’26
                - button "Delaware, regular election, Safe, projected margin D +28.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e195] [cursor=pointer]:
                  - generic [ref=e196]: DE
                  - generic [ref=e197]: ’26
                - button "Colorado, regular election, Safe, projected margin D +25.4 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations." [ref=e198] [cursor=pointer]:
                  - generic [ref=e199]: CO
                  - generic [ref=e200]: ’26
                - button "New Jersey, regular election, Safe, projected margin D +20.5 pp. At the D control threshold in 0.5% of simulations. At the R control threshold in 0.5% of simulations." [ref=e201] [cursor=pointer]:
                  - generic [ref=e202]: NJ
                  - generic [ref=e203]: ’26
                - button "Virginia, regular election, Safe, projected margin D +20.4 pp. At the D control threshold in 0.5% of simulations. At the R control threshold in 0.5% of simulations." [ref=e204] [cursor=pointer]:
                  - generic [ref=e205]: VA
                  - generic [ref=e206]: ’26
                - button "New Mexico, regular election, Safe, projected margin D +17.7 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e207] [cursor=pointer]:
                  - generic [ref=e208]: NM
                  - generic [ref=e209]: ’26
                - button "Illinois, regular election, Safe, projected margin D +16.5 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e210] [cursor=pointer]:
                  - generic [ref=e211]: IL
                  - generic [ref=e212]: ’26
                - button "Georgia, regular election, Safe, projected margin D +8.2 pp. At the D control threshold in 1.6% of simulations. At the R control threshold in 1.7% of simulations." [ref=e213] [cursor=pointer]:
                  - generic [ref=e214]: GA
                  - generic [ref=e215]: ’26
                - button "North Carolina, regular election, Safe, projected margin D +8.1 pp. At the D control threshold in 1.3% of simulations. At the R control threshold in 1.4% of simulations." [ref=e216] [cursor=pointer]:
                  - generic [ref=e217]: NC
                  - generic [ref=e218]: ’26
                - button "New Hampshire, regular election, Safe, projected margin D +7.7 pp. At the D control threshold in 2.1% of simulations. At the R control threshold in 2.1% of simulations." [ref=e219] [cursor=pointer]:
                  - generic [ref=e220]: NH
                  - generic [ref=e221]: ’26
                - button "Minnesota, regular election, Likely, projected margin D +4.8 pp. At the D control threshold in 5.2% of simulations. At the R control threshold in 5.2% of simulations." [ref=e222] [cursor=pointer]:
                  - generic [ref=e223]: MN
                  - generic [ref=e224]: ’26
                - button "Michigan, regular election, Likely, projected margin D +2.6 pp. At the D control threshold in 8.9% of simulations. At the R control threshold in 8.6% of simulations." [ref=e225] [cursor=pointer]:
                  - generic [ref=e226]: MI
                  - generic [ref=e227]: ’26
                - button "Maine, regular election, Likely, projected margin D +1.8 pp. At the D control threshold in 9.9% of simulations. At the R control threshold in 9.3% of simulations." [ref=e228] [cursor=pointer]:
                  - generic [ref=e229]: ME
                  - generic [ref=e230]: ’26
                - button "Ohio, special election, Tossup, projected margin R +0.3 pp. At the D control threshold in 13.1% of simulations. At the R control threshold in 11.9% of simulations." [pressed] [ref=e231] [cursor=pointer]:
                  - generic [ref=e232]: OH
                  - generic [ref=e233]: SP
                - button "Texas, regular election, Tossup, projected margin R +1.1 pp, projected D control threshold, projected R control threshold. At the D control threshold in 13.0% of simulations. At the R control threshold in 11.5% of simulations." [ref=e234] [cursor=pointer]:
                  - generic [ref=e236]: D51 / R50 · TX
                  - generic [ref=e237]: TX
                  - generic [ref=e238]: ’26
                - button "Alaska, regular election, Likely, projected margin R +2.1 pp. At the D control threshold in 10.1% of simulations. At the R control threshold in 8.8% of simulations." [ref=e239] [cursor=pointer]:
                  - generic [ref=e240]: AK
                  - generic [ref=e241]: ’26
                - button "Iowa, regular election, Likely, projected margin R +2.5 pp. At the D control threshold in 11.4% of simulations. At the R control threshold in 9.8% of simulations." [ref=e242] [cursor=pointer]:
                  - generic [ref=e243]: IA
                  - generic [ref=e244]: ’26
                - button "Kansas, regular election, Likely, projected margin R +3.7 pp. At the D control threshold in 8.7% of simulations. At the R control threshold in 7.4% of simulations." [ref=e245] [cursor=pointer]:
                  - generic [ref=e246]: KS
                  - generic [ref=e247]: ’26
                - button "South Carolina, regular election, Likely, projected margin R +5.4 pp. At the D control threshold in 6.0% of simulations. At the R control threshold in 5.0% of simulations." [ref=e248] [cursor=pointer]:
                  - generic [ref=e249]: SC
                  - generic [ref=e250]: ’26
                - button "Nebraska, regular election, Likely, projected margin R +5.7 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 4.7% of simulations." [ref=e251] [cursor=pointer]:
                  - generic [ref=e252]: NE
                  - generic [ref=e253]: ’26
                - button "Florida, special election, Safe, projected margin R +8.1 pp. At the D control threshold in 2.3% of simulations. At the R control threshold in 1.8% of simulations." [ref=e254] [cursor=pointer]:
                  - generic [ref=e255]: FL
                  - generic [ref=e256]: SP
                - button "Montana, regular election, Safe, projected margin R +8.3 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.4% of simulations." [ref=e257] [cursor=pointer]:
                  - generic [ref=e258]: MT
                  - generic [ref=e259]: ’26
                - button "Idaho, regular election, Likely, projected margin R +8.8 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.8% of simulations." [ref=e260] [cursor=pointer]:
                  - generic [ref=e261]: ID
                  - generic [ref=e262]: ’26
                - button "Louisiana, regular election, Likely, projected margin R +12.0 pp. At the D control threshold in 2.3% of simulations. At the R control threshold in 2.0% of simulations." [ref=e263] [cursor=pointer]:
                  - generic [ref=e264]: LA
                  - generic [ref=e265]: ’26
                - button "Mississippi, regular election, Safe, projected margin R +13.4 pp. At the D control threshold in 1.1% of simulations. At the R control threshold in 0.9% of simulations." [ref=e266] [cursor=pointer]:
                  - generic [ref=e267]: MS
                  - generic [ref=e268]: ’26
                - button "Arkansas, regular election, Safe, projected margin R +17.1 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.2% of simulations." [ref=e269] [cursor=pointer]:
                  - generic [ref=e270]: AR
                  - generic [ref=e271]: ’26
                - button "South Dakota, regular election, Safe, projected margin R +17.9 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e272] [cursor=pointer]:
                  - generic [ref=e273]: SD
                  - generic [ref=e274]: ’26
                - button "Kentucky, regular election, Safe, projected margin R +21.0 pp. At the D control threshold in 0.6% of simulations. At the R control threshold in 0.5% of simulations." [ref=e275] [cursor=pointer]:
                  - generic [ref=e276]: KY
                  - generic [ref=e277]: ’26
                - button "Alabama, regular election, Safe, projected margin R +21.5 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e278] [cursor=pointer]:
                  - generic [ref=e279]: AL
                  - generic [ref=e280]: ’26
                - button "Oklahoma, regular election, Safe, projected margin R +25.1 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.3% of simulations." [ref=e281] [cursor=pointer]:
                  - generic [ref=e282]: OK
                  - generic [ref=e283]: ’26
                - button "Tennessee, regular election, Safe, projected margin R +26.2 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations." [ref=e284] [cursor=pointer]:
                  - generic [ref=e285]: TN
                  - generic [ref=e286]: ’26
                - button "West Virginia, regular election, Safe, projected margin R +37.2 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e287] [cursor=pointer]:
                  - generic [ref=e288]: WV
                  - generic [ref=e289]: ’26
                - button "Wyoming, regular election, Safe, projected margin R +37.9 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.0% of simulations." [ref=e290] [cursor=pointer]:
                  - generic [ref=e291]: WY
                  - generic [ref=e292]: ’26
                - generic [ref=e293]:
                  - strong [ref=e294]: "31"
                  - generic [ref=e295]: R caucusnot up this year
            - paragraph [ref=e296]: Read inward from either party’s end. Its marker identifies the last race needed for control in this projected order, after seats not up for election and races guaranteed to join that party’s group. D = Democratic; R = Republican. “SP” marks a special election to fill an unfinished term.
            - generic [ref=e297]:
              - generic [ref=e298]:
                - generic [ref=e299]: D
                - generic [ref=e300]:
                  - strong [ref=e301]:
                    - button "Texas" [ref=e302] [cursor=pointer]
                  - generic [ref=e303]: 51 caucus seats for control · projected margin −1.1 pp
                  - generic [ref=e304]:
                    - text: "Most frequent in simulations:"
                    - button "Ohio" [ref=e305] [cursor=pointer]
                    - text: · 13.1%
              - generic [ref=e306]:
                - generic [ref=e307]: R
                - generic [ref=e308]:
                  - strong [ref=e309]:
                    - button "Texas" [ref=e310] [cursor=pointer]
                  - generic [ref=e311]: 50 caucus seats for control · projected margin +1.1 pp
                  - generic [ref=e312]:
                    - text: "Most frequent in simulations:"
                    - button "Ohio" [ref=e313] [cursor=pointer]
                    - text: · 11.9%
            - generic [ref=e314]:
              - strong [ref=e315]: Ohio · R +0.3 pp
              - generic [ref=e316]: At the D control threshold in 13.1% of simulations.
              - generic [ref=e317]: At the R control threshold in 11.9% of simulations.
            - paragraph [ref=e318]: "A margin is the difference in vote shares: 52% versus 48% is a 4 percentage-point (pp) lead. The frequencies above count how often a race occupies the control-threshold rank across simulated elections, including simulations where that party falls short. They are not the chance that flipping that race alone would change control."
            - group [ref=e319]:
              - generic "How to read the thresholds" [ref=e320] [cursor=pointer]
        - region "Senate race explorer" [ref=e321]:
          - generic [ref=e322]:
            - generic [ref=e323]:
              - generic [ref=e324]:
                - generic [ref=e325]:
                  - heading "A nation of distinct races" [level=2] [ref=e326]
                  - generic [ref=e327]: Select a state to explore its election.
                - generic [ref=e328]: 2026 Senate
              - group "United States Senate races. Use Tab to select a state with an election." [ref=e329]:
                - button "New Mexico, 1 election, Safe" [ref=e330] [cursor=pointer]
                - button "South Dakota, 1 election, Safe" [ref=e331] [cursor=pointer]
                - generic "California, no Senate election" [ref=e332]
                - button "Kentucky, 1 election, Safe" [ref=e333] [cursor=pointer]
                - button "Alabama, 1 election, Safe" [ref=e334] [cursor=pointer]
                - button "Georgia, 1 election, Safe" [ref=e335] [cursor=pointer]
                - button "Arkansas, 1 election, Safe" [ref=e336] [cursor=pointer]
                - generic "Pennsylvania, no Senate election" [ref=e337]
                - generic "Missouri, no Senate election" [ref=e338]
                - button "Colorado, 1 election, Safe" [ref=e339] [cursor=pointer]
                - generic "Utah, no Senate election" [ref=e340]
                - button "Oklahoma, 1 election, Safe" [ref=e341] [cursor=pointer]
                - button "Tennessee, 1 election, Safe" [ref=e342] [cursor=pointer]
                - button "Wyoming, 1 election, Safe" [ref=e343] [cursor=pointer]
                - generic "New York, no Senate election" [ref=e344]
                - generic "Indiana, no Senate election" [ref=e345]
                - button "Kansas, 1 election, Likely" [ref=e346] [cursor=pointer]
                - button "Idaho, 1 election, Likely" [ref=e347] [cursor=pointer]
                - button "Alaska, 1 election, Likely" [ref=e348] [cursor=pointer]
                - generic "Nevada, no Senate election" [ref=e349]
                - button "Illinois, 1 election, Safe" [ref=e350] [cursor=pointer]
                - generic "Vermont, no Senate election" [ref=e351]
                - button "New Jersey, 1 election, Safe" [ref=e352] [cursor=pointer]
                - generic "North Dakota, no Senate election" [ref=e353]
                - button "Iowa, 1 election, Likely" [ref=e354] [cursor=pointer]
                - button "South Carolina, 1 election, Likely" [ref=e355] [cursor=pointer]
                - generic "Arizona, no Senate election" [ref=e356]
                - button "Delaware, 1 election, Safe" [ref=e357] [cursor=pointer]
                - generic "District of Columbia, no Senate election" [ref=e358]
                - generic "Connecticut, no Senate election" [ref=e359]
                - button "New Hampshire, 1 election, Safe" [ref=e360] [cursor=pointer]
                - button "Nebraska, 1 election, Likely" [ref=e361] [cursor=pointer]
                - generic "Washington, no Senate election" [ref=e362]
                - button "Texas, 1 election, Tossup" [ref=e363] [cursor=pointer]
                - button "Ohio, 1 election, Tossup" [pressed] [ref=e364] [cursor=pointer]
                - generic "Wisconsin, no Senate election" [ref=e365]
                - button "Oregon, 1 election, Safe" [ref=e366] [cursor=pointer]
                - button "Mississippi, 1 election, Safe" [ref=e367] [cursor=pointer]
                - button "North Carolina, 1 election, Safe" [ref=e368] [cursor=pointer]
                - button "Virginia, 1 election, Safe" [ref=e369] [cursor=pointer]
                - button "West Virginia, 1 election, Safe" [ref=e370] [cursor=pointer]
                - button "Louisiana, 1 election, Likely" [ref=e371] [cursor=pointer]
                - button "Michigan, 1 election, Likely" [ref=e372] [cursor=pointer]
                - button "Massachusetts, 1 election, Safe" [ref=e373] [cursor=pointer]
                - button "Florida, 1 election, Safe" [ref=e374] [cursor=pointer]
                - button "Montana, 1 election, Safe" [ref=e375] [cursor=pointer]
                - button "Minnesota, 1 election, Likely" [ref=e376] [cursor=pointer]
                - generic "Maryland, no Senate election" [ref=e377]
                - button "Maine, 1 election, Likely" [ref=e378] [cursor=pointer]
                - generic "Hawaii, no Senate election" [ref=e379]
                - button "Rhode Island, 1 election, Safe" [ref=e380] [cursor=pointer]
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
                - generic [ref=e381]: Alaska and Hawaii shown in insets; scale varies.
              - generic [ref=e382]:
                - generic [ref=e383]: Safe D
                - generic [ref=e385]: Likely D
                - generic [ref=e387]: Safe R
                - generic [ref=e389]: Likely R
                - generic [ref=e391]: Safe I
                - generic [ref=e393]: Likely I
                - generic [ref=e395]: Tossup
              - generic [ref=e397]:
                - generic [ref=e398]: "Ratings use the favorite’s chance of winning: Safe above 90%; Likely above 60% through 90%; Tossup 60% or lower (40–60% each in a two-outcome race). Safe does not mean certain."
                - generic [ref=e399]: "D = Democratic · R = Republican · I = Independent. Gray: no election. State area does not represent votes."
            - region [ref=e400]:
              - generic [ref=e402]:
                - heading "Who could control the chamber?" [level=2] [ref=e403]
                - generic [ref=e404]: Independents unaligned · D needs 51; R needs 50
              - generic [ref=e405]:
                - generic [ref=e406]:
                  - generic [ref=e407]: Democratic control
                  - generic [ref=e409]: 43.5%
                - generic [ref=e410]:
                  - generic [ref=e411]: Republican control
                  - generic [ref=e413]: 53.2%
                - generic [ref=e414]:
                  - generic [ref=e415]: Control depends on alliances
                  - generic [ref=e417]: 3.3%
              - paragraph [ref=e422]: King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
              - paragraph [ref=e423]: A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.
              - generic [ref=e424]:
                - generic [ref=e425]:
                  - text: "D caucus: 50.1 expected seats"
                  - generic [ref=e426]: (46–55, modeled 90% range)
                - generic [ref=e427]:
                  - text: "R caucus: 49.6 expected seats"
                  - generic [ref=e428]: (44–54, modeled 90% range)
                - generic [ref=e429]:
                  - text: "Unaligned: 0.3 expected seats"
                  - generic [ref=e430]: (0–2, modeled 90% range)
              - paragraph [ref=e431]: Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.
            - region [ref=e432]:
              - heading "What has been tested?" [level=2] [ref=e433]
              - paragraph [ref=e434]: The 2026 election has not happened yet. Select 2018 or 2022 to see how historical reconstructions compare with actual results. Those views use a different model of possible winners, so their scores do not measure the accuracy of this 2026 two-contender forecast.
              - group [ref=e435]:
                - generic "Scope and limitations" [ref=e436] [cursor=pointer]
            - region [ref=e437]:
              - generic [ref=e438]:
                - heading "The races" [level=2] [ref=e439]
                - generic [ref=e440]: 10 of 35 elections
              - paragraph [ref=e441]: The favorite is the candidate or group with the highest modeled chance of winning. “Most competitive” puts the least certain favorites first.
              - generic [ref=e442]:
                - generic [ref=e443]:
                  - generic [aria-hidden] [ref=e444]: ⌕
                  - textbox "Search states or candidates" [ref=e445]:
                    - /placeholder: Find a state or candidate
                - combobox "Sort races" [ref=e446]:
                  - option "Most competitive" [selected]
                  - option "State A–Z"
              - table [ref=e447]:
                - rowgroup [ref=e448]:
                  - row [ref=e449]:
                    - columnheader "State / election" [ref=e450]
                    - columnheader "Favorite" [ref=e451]
                    - columnheader "Chance to win" [ref=e452]
                - rowgroup [ref=e453]:
                  - row [ref=e454]:
                    - cell [ref=e455]:
                      - button "Ohio Special election" [pressed] [ref=e456] [cursor=pointer]:
                        - text: Ohio
                        - generic [ref=e457]: Special election
                    - cell "Jon Husted Republican" [ref=e458]:
                      - text: Jon Husted
                      - generic [ref=e459]: Republican
                    - cell "52.6%" [ref=e460]
                  - row [ref=e464]:
                    - cell [ref=e465]:
                      - button "Texas Regular election" [ref=e466] [cursor=pointer]:
                        - text: Texas
                        - generic [ref=e467]: Regular election
                    - cell "KEN PAXTON Republican" [ref=e468]:
                      - text: KEN PAXTON
                      - generic [ref=e469]: Republican
                    - cell "58.6%" [ref=e470]
                  - row [ref=e474]:
                    - cell [ref=e475]:
                      - button "Maine Regular election" [ref=e476] [cursor=pointer]:
                        - text: Maine
                        - generic [ref=e477]: Regular election
                    - cell "Jackson, Troy D. Democratic" [ref=e478]:
                      - text: Jackson, Troy D.
                      - generic [ref=e479]: Democratic
                    - cell "63.0%" [ref=e480]
                  - row [ref=e484]:
                    - cell [ref=e485]:
                      - button "Alaska Regular election" [ref=e486] [cursor=pointer]:
                        - text: Alaska
                        - generic [ref=e487]: Regular election
                    - cell "Sullivan, Dan S. Republican" [ref=e488]:
                      - text: Sullivan, Dan S.
                      - generic [ref=e489]: Republican
                    - cell "64.4%" [ref=e490]
                  - row [ref=e494]:
                    - cell [ref=e495]:
                      - button "Iowa Regular election" [ref=e496] [cursor=pointer]:
                        - text: Iowa
                        - generic [ref=e497]: Regular election
                    - cell "Ashley Hinson Republican" [ref=e498]:
                      - text: Ashley Hinson
                      - generic [ref=e499]: Republican
                    - cell "68.6%" [ref=e500]
                  - row [ref=e504]:
                    - cell [ref=e505]:
                      - button "Michigan Regular election" [ref=e506] [cursor=pointer]:
                        - text: Michigan
                        - generic [ref=e507]: Regular election
                    - cell "Abdul El-Sayed Democratic" [ref=e508]:
                      - text: Abdul El-Sayed
                      - generic [ref=e509]: Democratic
                    - cell "68.8%" [ref=e510]
                  - row [ref=e514]:
                    - cell [ref=e515]:
                      - button "Kansas Regular election" [ref=e516] [cursor=pointer]:
                        - text: Kansas
                        - generic [ref=e517]: Regular election
                    - cell "Roger Marshall Republican" [ref=e518]:
                      - text: Roger Marshall
                      - generic [ref=e519]: Republican
                    - cell "74.5%" [ref=e520]
                  - row [ref=e524]:
                    - cell [ref=e525]:
                      - button "Minnesota Regular election" [ref=e526] [cursor=pointer]:
                        - text: Minnesota
                        - generic [ref=e527]: Regular election
                    - cell "Peggy Flanagan Democratic" [ref=e528]:
                      - text: Peggy Flanagan
                      - generic [ref=e529]: Democratic
                    - cell "80.3%" [ref=e530]
                  - row [ref=e534]:
                    - cell [ref=e535]:
                      - button "South Carolina Regular election" [ref=e536] [cursor=pointer]:
                        - text: South Carolina
                        - generic [ref=e537]: Regular election
                    - cell "Darline Graham Republican" [ref=e538]:
                      - text: Darline Graham
                      - generic [ref=e539]: Republican
                    - cell "82.7%" [ref=e540]
                  - row [ref=e544]:
                    - cell [ref=e545]:
                      - button "Nebraska Regular election" [ref=e546] [cursor=pointer]:
                        - text: Nebraska
                        - generic [ref=e547]: Regular election
                    - cell "Pete Ricketts Republican" [ref=e548]:
                      - text: Pete Ricketts
                      - generic [ref=e549]: Republican
                    - cell "84.4%" [ref=e550]
              - button "See all 35 races" [ref=e554] [cursor=pointer]:
                - text: See all 35 races
                - generic [aria-hidden] [ref=e555]: ⌄
          - complementary "Selected race detail" [ref=e556]:
            - generic [ref=e557]:
              - generic [ref=e558]: Race in focus
              - generic [ref=e559]: Tossup
            - heading "Ohio" [level=2] [ref=e560]
            - generic [ref=e561]: U.S. Senate · Special election
            - separator [ref=e562]
            - heading "Chance of winning" [level=3] [ref=e563]
            - paragraph [ref=e564]: A 70% chance means winning about 70 out of 100 elections simulated by this model. It does not mean winning 70% of the votes.
            - generic [ref=e565]:
              - generic [ref=e567]:
                - generic [ref=e568]:
                  - generic [ref=e569]: Sherrod Brown
                  - generic [ref=e570]: Democratic
                - strong [ref=e571]: 47.4%
              - generic [ref=e575]:
                - generic [ref=e576]:
                  - generic [ref=e577]: Jon Husted
                  - generic [ref=e578]: Republican · Currently holds this seat
                - strong [ref=e579]: 52.6%
            - separator [ref=e582]
            - heading "Vote share among these two contenders" [level=3] [ref=e583]
            - generic [ref=e584]:
              - generic [ref=e585]:
                - generic [ref=e586]:
                  - generic [ref=e587]: Brown
                  - strong [ref=e588]: 49.8%
                - 'generic "Sherrod Brown: mean 49.8%, Modeled 90% range 45.6% to 54.2%" [ref=e589]'
                - paragraph [ref=e592]: "Modeled 90% range: 45.6%–54.2%Share of votes for the two named principal contenders"
              - generic [ref=e593]:
                - generic [ref=e594]:
                  - generic [ref=e595]: Husted
                  - strong [ref=e596]: 50.2%
                - 'generic "Jon Husted: mean 50.2%, Modeled 90% range 45.8% to 54.4%" [ref=e597]'
                - paragraph [ref=e600]: "Modeled 90% range: 45.8%–54.4%Share of votes for the two named principal contenders"
            - paragraph [ref=e601]: The shares divide votes between these two contenders only; votes for other candidates are excluded. The range runs from the 5th to the 95th percentile of simulated shares—the middle 90%. It is a model estimate, not a poll’s margin of error or a guarantee.
            - paragraph [ref=e602]: Probabilities of 0% or 100% reflect the saved simulation, not certainty.
            - separator [ref=e603]
            - generic [ref=e604]:
              - generic [ref=e605]:
                - text: Selected model setup
                - strong [ref=e606]: Standard polling
              - generic [ref=e607]:
                - text: Poll records included
                - strong [ref=e608]: "11"
            - paragraph [ref=e609]:
              - text: This counts poll records used for this race, not people interviewed or every poll available.
              - link "Why some polls are included—and others are not ↓" [ref=e610] [cursor=pointer]:
                - /url: "#poll-selection"
            - group [ref=e611]:
              - generic "Race assumptions" [ref=e612] [cursor=pointer]
            - separator [ref=e613]
            - heading "Candidate and result sources" [level=3] [ref=e614]
            - generic [ref=e615]:
              - link "Sherrod Brown ↗" [ref=e617] [cursor=pointer]:
                - /url: https://www.boe.ohio.gov/abmt/oh12/20261103/X001___2_CITY_1.html
              - link "Jon Husted ↗" [ref=e619] [cursor=pointer]:
                - /url: https://www.boe.ohio.gov/abmt/oh12/20261103/X001___2_CITY_1.html
            - paragraph [ref=e620]: "Forecast date: October 26, 2026Inputs dated through: October 26, 2026Saved research forecast; new polls are not added automatically."
        - region [ref=e621]:
          - generic [ref=e622]:
            - generic [ref=e623]: From evidence to possible elections
            - heading "How this forecast works." [level=2] [ref=e624]
            - paragraph [ref=e625]: A model is a set of mathematical assumptions connecting evidence to possible results. Your choices change those assumptions and, when available, load the matching saved forecast.
            - link "Download this forecast’s data ↓" [ref=e626] [cursor=pointer]:
              - /url: /data/explorer/current-polling.json
            - paragraph [ref=e627]: "The download is a JSON file: structured data for analysis, with source identifiers and limitations."
          - generic [ref=e628]:
            - generic [ref=e629]:
              - heading "1. Start with a baseline" [level=3] [ref=e630]
              - paragraph [ref=e631]: Electoral history supplies a starting estimate, often called “fundamentals.” The builder also offers a historical winner reference model. A component is one part of the model; the chosen combination is your model setup or “stack.”
            - generic [ref=e632]:
              - heading "2. Add the selected evidence" [level=3] [ref=e633]
              - paragraph [ref=e634]:
                - text: This setup adds the polling evidence saved for this election. Polls measure responses at a particular time. The model allows for measurement errors and changes before election day.
                - link "Read the poll selection rules below." [ref=e635] [cursor=pointer]:
                  - /url: "#poll-selection"
            - generic [ref=e636]:
              - heading "3. Simulate elections together" [level=3] [ref=e637]
              - paragraph [ref=e638]: A simulation is one possible set of election results generated by the model. Shared uncertainty allows many races to move together—for example, if polls miss in the same direction. Win chances and chamber control summarize these simulated elections, rather than assuming each race is unrelated.
            - generic [ref=e639]:
              - heading "4. Explore what changes" [level=3] [ref=e640]
              - paragraph [ref=e641]: Changing a choice loads a saved result if one exists. Some combinations require calculating several model parts together again (a “joint refit”); the website explains when that work is needed. A sensitivity test changes assumptions without adding evidence. “Runs successfully,” “experimental,” and “tested against election results” describe different things.
            - generic [ref=e642]:
              - heading "What these numbers cover" [level=3] [ref=e643]
              - paragraph [ref=e644]: The 2026 preview models only the two named contenders in each race. Its vote shares divide votes between those two; other candidates’ shares and chances of winning are outside this model.
            - generic [ref=e645]:
              - heading "Make fair comparisons" [level=3] [ref=e646]
              - paragraph [ref=e647]: Compare the same year, possible winners and assumptions about independent senators. A change from 40% to 45% is 5 percentage points (“pp”), not a 5% relative increase. Historical scores describe those elections and do not establish the accuracy of the current forecast.
        - region [ref=e648]:
          - generic [ref=e649]: The evidence behind this view
          - heading "Why some polls are included—and others are not" [level=2] [ref=e650]
          - paragraph [ref=e651]:
            - text: This saved forecast uses
            - strong [ref=e652]: 131 poll records across 25 of 35 races
            - text: ", from sources collected by October 26, 2026. A record is one selected survey question for one race. It is not a count of people interviewed, and this collection does not include every published poll."
          - generic [ref=e653]:
            - generic [ref=e654]:
              - heading "Match the contest and the date" [level=3] [ref=e655]
              - paragraph [ref=e656]: The race and named contenders must match the forecast. Sources must document usable percentages and show that the evidence was available by the saved cutoff. Fieldwork must end within 100 days of election day. An availability check does not establish the exact day a poll was published.
            - generic [ref=e657]:
              - heading "Avoid counting the same evidence twice" [level=3] [ref=e658]
              - paragraph [ref=e659]: Only one question is chosen from a survey, so alternative ballot questions do not become extra polls. For each race, the latest eligible sample in each polling-provider grouping is retained. Where the actual survey contractor is unknown, the named survey organization supplies that grouping; this is an approximation, not a verified contractor identity.
            - generic [ref=e660]:
              - heading "Keep source gaps visible" [level=3] [ref=e661]
              - paragraph [ref=e662]: Missing or uncertain source documents, dates or suitable questions can keep a record out. An absent poll may simply be outside the collected sources. Its absence is not, by itself, a judgment that the pollster is unreliable.
          - group [ref=e663]:
            - generic "How sample sizes, voter types and missing polls are handled" [ref=e664] [cursor=pointer]
          - group [ref=e665]:
            - generic "Check included records and recorded exclusions by state" [ref=e666] [cursor=pointer]
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
        - generic [ref=e667]:
          - generic [ref=e668]: ballotmancy
          - generic [ref=e669]: Independent research. Visible assumptions.2026 Senate research preview · October 26, 2026
  - alert [ref=e670]
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