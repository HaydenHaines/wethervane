# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: election-night.spec.ts >> senate: same-page guide starts standard, renders every exact model, and preserves the visitor's forecast
- Location: tests/site/election-night.spec.ts:10:7

# Error details

```
Error: locator.click: Error: strict mode violation: locator('#election-night-scorecard').locator('.night-all-races summary') resolved to 11 elements:
    1) <summary>All 35 predictions</summary> aka getByText('All 35 predictions', { exact: true })
    2) <summary>Prediction details and what is covered</summary> aka locator('.night-predictions > article > .night-race-definitions > summary').first()
    3) <summary>Prediction details and what is covered</summary> aka locator('.night-predictions > article:nth-child(2) > .night-race-definitions > summary')
    4) <summary>Prediction details and what is covered</summary> aka locator('.night-predictions > article:nth-child(3) > .night-race-definitions > summary')
    5) <summary>Prediction details and what is covered</summary> aka locator('.night-predictions > article:nth-child(4) > .night-race-definitions > summary')
    6) <summary>Prediction details and what is covered</summary> aka locator('.night-predictions > article:nth-child(5) > .night-race-definitions > summary')
    7) <summary>Prediction details and what is covered</summary> aka locator('.night-predictions > article:nth-child(6) > .night-race-definitions > summary')
    8) <summary>Prediction details and what is covered</summary> aka locator('article:nth-child(7) > .night-race-definitions > summary')
    9) <summary>Prediction details and what is covered</summary> aka locator('article:nth-child(8) > .night-race-definitions > summary')
    10) <summary>Prediction details and what is covered</summary> aka locator('article:nth-child(9) > .night-race-definitions > summary')
    ...

Call log:
  - waiting for locator('#election-night-scorecard').locator('.night-all-races summary')

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
        - button "Election night scorecard" [expanded] [ref=e8] [cursor=pointer]
        - generic [ref=e9]: Research preview
    - generic [ref=e11]:
      - tablist "Chamber" [ref=e12]:
        - tab "Senate" [selected] [ref=e13] [cursor=pointer]
        - tab "House" [ref=e14] [cursor=pointer]
      - generic [ref=e15]: Two chambers. Your perspective.
    - region [ref=e16]:
      - generic [ref=e17]:
        - generic [ref=e18]:
          - generic [ref=e19]: An election-night watcher’s guide
          - heading "Election night scorecard" [level=2] [ref=e20]
          - paragraph [ref=e21]: What the model predicts, how wrong it expects to be, and what the results can tell us.
        - button "Close scorecard" [ref=e22] [cursor=pointer]
      - generic [ref=e23]:
        - strong [ref=e24]: Senate · Standard polling
        - generic [ref=e25]: Standard forecast · Dated snapshot
        - paragraph [ref=e26]: Inputs through Tue, 20 Oct 2026 16:21:44 GMT. Forecast issued Tue, 20 Oct 2026 16:27:31 GMT.
        - paragraph [ref=e27]: This is a saved forecast edition. The final election-night evaluation edition and grading rules have not yet been locked.
      - group [ref=e28]:
        - generic "Inspect other model stacks" [active] [ref=e29] [cursor=pointer]
        - generic [ref=e30]:
          - text: Scorecard model
          - combobox "Scorecard model" [ref=e31]:
            - option "Recent calendar history · without polls"
            - option "Recent calendar history · with polls"
            - option "Refreshed history · without polls"
            - option "Refreshed history · with polls"
            - option "Learned presidential slope · without polls"
            - option "Learned presidential slope · with polls"
            - option "Relative lean with presidential context · without polls"
            - option "Relative lean with presidential context · with polls"
            - option "Relative state and national lean · without polls"
            - option "Relative state and national lean · with polls"
            - 'option "Polling error: common-cycle SD × 2"'
            - 'option "Polling error: common-cycle SD × 0.5"'
            - 'option "Polling error: all scales × 1.5"'
            - 'option "Polling error: all scales × 2/3"'
            - option "Fundamentals"
            - option "Standard polling · Standard" [selected]
        - paragraph [ref=e32]: Changes here affect only this scorecard. Opening it again starts with the standard forecast.
      - region [ref=e33]:
        - generic [ref=e34]: 1 · Results and predictive power
        - heading "The predictions we can check" [level=3] [ref=e35]
        - generic [ref=e36]:
          - generic [ref=e37]:
            - strong [ref=e38]: "35"
            - generic [ref=e39]: modeled contests
          - generic [ref=e40]:
            - strong [ref=e41]: "35"
            - generic [ref=e42]: with a two-candidate margin
          - generic [ref=e43]:
            - strong [ref=e44]: Not scored
            - generic [ref=e45]: verified election results have not been added
        - heading "Races to watch" [level=4] [ref=e46]
        - paragraph [ref=e47]: These six examples have the least certain winners in the dated standard forecast. The same races stay here when you inspect another stack. Their margins show whether an even win chance also means a close expected race.
        - generic [ref=e48]:
          - article [ref=e49]:
            - generic [ref=e50]:
              - heading "Ohio · Special election" [level=4] [ref=e51]
              - generic [ref=e52]: Awaiting verified result
            - generic [ref=e53]:
              - generic [ref=e54]:
                - term [ref=e55]: Predicted two-candidate margin
                - definition [ref=e56]: Jon Husted +0.6 points
              - generic [ref=e57]:
                - term [ref=e58]: Highest win chance
                - definition [ref=e59]: Jon Husted · 54.4%
              - generic [ref=e60]:
                - term [ref=e61]: Final result / prediction error
                - definition [ref=e62]: Not scored
            - generic [ref=e63]:
              - img "Average margin −0.6 points; middle 90% from −10.0 to +8.9 points. Positive means Sherrod Brown ahead." [ref=e64]:
                - generic [ref=e66]: −20
                - generic [ref=e67]: "0"
                - generic [ref=e68]: +20 points
              - paragraph [ref=e69]:
                - text: "Middle 90%:"
                - strong [ref=e70]: −10.0 to +8.9 points
                - text: .
              - paragraph [ref=e71]: "Positive means Sherrod Brown ahead; negative means Jon Husted ahead. Dot: average margin."
            - group [ref=e72]:
              - generic "Prediction details and what is covered" [ref=e73] [cursor=pointer]
          - article [ref=e74]:
            - generic [ref=e75]:
              - heading "Texas · Regular election" [level=4] [ref=e76]
              - generic [ref=e77]: Awaiting verified result
            - generic [ref=e78]:
              - generic [ref=e79]:
                - term [ref=e80]: Predicted two-candidate margin
                - definition [ref=e81]: KEN PAXTON +1.1 points
              - generic [ref=e82]:
                - term [ref=e83]: Highest win chance
                - definition [ref=e84]: KEN PAXTON · 58.2%
              - generic [ref=e85]:
                - term [ref=e86]: Final result / prediction error
                - definition [ref=e87]: Not scored
            - generic [ref=e88]:
              - img "Average margin −1.1 points; middle 90% from −10.2 to +8.1 points. Positive means JAMES TALARICO ahead." [ref=e89]:
                - generic [ref=e91]: −20
                - generic [ref=e92]: "0"
                - generic [ref=e93]: +20 points
              - paragraph [ref=e94]:
                - text: "Middle 90%:"
                - strong [ref=e95]: −10.2 to +8.1 points
                - text: .
              - paragraph [ref=e96]: "Positive means JAMES TALARICO ahead; negative means KEN PAXTON ahead. Dot: average margin."
            - group [ref=e97]:
              - generic "Prediction details and what is covered" [ref=e98] [cursor=pointer]
          - article [ref=e99]:
            - generic [ref=e100]:
              - heading "Alaska · Regular election" [level=4] [ref=e101]
              - generic [ref=e102]: Awaiting verified result
            - generic [ref=e103]:
              - generic [ref=e104]:
                - term [ref=e105]: Predicted two-candidate margin
                - definition [ref=e106]: Sullivan, Dan S. +1.7 points
              - generic [ref=e107]:
                - term [ref=e108]: Highest win chance
                - definition [ref=e109]: Sullivan, Dan S. · 61.0%
              - generic [ref=e110]:
                - term [ref=e111]: Final result / prediction error
                - definition [ref=e112]: Not scored
            - generic [ref=e113]:
              - img "Average margin −1.7 points; middle 90% from −12.2 to +9.1 points. Positive means Peltola, Mary ahead." [ref=e114]:
                - generic [ref=e116]: −20
                - generic [ref=e117]: "0"
                - generic [ref=e118]: +20 points
              - paragraph [ref=e119]:
                - text: "Middle 90%:"
                - strong [ref=e120]: −12.2 to +9.1 points
                - text: .
              - paragraph [ref=e121]: "Positive means Peltola, Mary ahead; negative means Sullivan, Dan S. ahead. Dot: average margin."
            - group [ref=e122]:
              - generic "Prediction details and what is covered" [ref=e123] [cursor=pointer]
          - article [ref=e124]:
            - generic [ref=e125]:
              - heading "Iowa · Regular election" [level=4] [ref=e126]
              - generic [ref=e127]: Awaiting verified result
            - generic [ref=e128]:
              - generic [ref=e129]:
                - term [ref=e130]: Predicted two-candidate margin
                - definition [ref=e131]: Ashley Hinson +2.3 points
              - generic [ref=e132]:
                - term [ref=e133]: Highest win chance
                - definition [ref=e134]: Ashley Hinson · 66.3%
              - generic [ref=e135]:
                - term [ref=e136]: Final result / prediction error
                - definition [ref=e137]: Not scored
            - generic [ref=e138]:
              - img "Average margin −2.3 points; middle 90% from −11.2 to +6.9 points. Positive means Josh Turek ahead." [ref=e139]:
                - generic [ref=e141]: −20
                - generic [ref=e142]: "0"
                - generic [ref=e143]: +20 points
              - paragraph [ref=e144]:
                - text: "Middle 90%:"
                - strong [ref=e145]: −11.2 to +6.9 points
                - text: .
              - paragraph [ref=e146]: "Positive means Josh Turek ahead; negative means Ashley Hinson ahead. Dot: average margin."
            - group [ref=e147]:
              - generic "Prediction details and what is covered" [ref=e148] [cursor=pointer]
          - article [ref=e149]:
            - generic [ref=e150]:
              - heading "Idaho · Regular election" [level=4] [ref=e151]
              - generic [ref=e152]: Awaiting verified result
            - generic [ref=e153]:
              - generic [ref=e154]:
                - term [ref=e155]: Predicted two-candidate margin
                - definition [ref=e156]: Jim Risch +4.2 points
              - generic [ref=e157]:
                - term [ref=e158]: Highest win chance
                - definition [ref=e159]: Jim Risch · 68.0%
              - generic [ref=e160]:
                - term [ref=e161]: Final result / prediction error
                - definition [ref=e162]: Not scored
            - generic [ref=e163]:
              - img "Average margin −4.2 points; middle 90% from −19.1 to +10.9 points. Positive means Todd Achilles ahead." [ref=e164]:
                - generic [ref=e166]: −20
                - generic [ref=e167]: "0"
                - generic [ref=e168]: +20 points
              - paragraph [ref=e169]:
                - text: "Middle 90%:"
                - strong [ref=e170]: −19.1 to +10.9 points
                - text: .
              - paragraph [ref=e171]: "Positive means Todd Achilles ahead; negative means Jim Risch ahead. Dot: average margin."
            - group [ref=e172]:
              - generic "Prediction details and what is covered" [ref=e173] [cursor=pointer]
          - article [ref=e174]:
            - generic [ref=e175]:
              - heading "Kansas · Regular election" [level=4] [ref=e176]
              - generic [ref=e177]: Awaiting verified result
            - generic [ref=e178]:
              - generic [ref=e179]:
                - term [ref=e180]: Predicted two-candidate margin
                - definition [ref=e181]: Roger Marshall +3.2 points
              - generic [ref=e182]:
                - term [ref=e183]: Highest win chance
                - definition [ref=e184]: Roger Marshall · 70.2%
              - generic [ref=e185]:
                - term [ref=e186]: Final result / prediction error
                - definition [ref=e187]: Not scored
            - generic [ref=e188]:
              - img "Average margin −3.2 points; middle 90% from −13.0 to +7.0 points. Positive means Adam Hamilton ahead." [ref=e189]:
                - generic [ref=e191]: −20
                - generic [ref=e192]: "0"
                - generic [ref=e193]: +20 points
              - paragraph [ref=e194]:
                - text: "Middle 90%:"
                - strong [ref=e195]: −13.0 to +7.0 points
                - text: .
              - paragraph [ref=e196]: "Positive means Adam Hamilton ahead; negative means Roger Marshall ahead. Dot: average margin."
            - group [ref=e197]:
              - generic "Prediction details and what is covered" [ref=e198] [cursor=pointer]
        - group [ref=e199]:
          - generic "All 35 predictions" [ref=e200] [cursor=pointer]
        - paragraph [ref=e201]: When verified results are added, a called winner and a final vote margin will be tracked separately. An incomplete count will not be scored as a final margin.
      - region [ref=e202]:
        - generic [ref=e203]: 2 · Uncertainty
        - heading "What the forecast expects to get wrong" [level=3] [ref=e204]
        - generic [ref=e205]:
          - article [ref=e206]:
            - heading "Wrong top choices" [level=4] [ref=e207]
            - generic [ref=e208]:
              - text: "4.2"
              - generic [ref=e209]: of 35
            - paragraph [ref=e210]: The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.
            - paragraph [ref=e211]: These choices cover the two named principal contenders under each race’s stated assumptions.
          - article [ref=e212]:
            - heading "How far the margins could miss" [level=4] [ref=e213]
            - paragraph [ref=e214]: Awaiting a reviewed uncertainty summary
            - paragraph [ref=e215]: The margin ranges above are available. The model’s expected average margin error and the range for the number of misses have not yet been published.
        - article [ref=e216]:
          - heading "Shared uncertainty in the chamber" [level=4] [ref=e217]
          - paragraph [ref=e218]: Independents unaligned. Existing King and Sanders remain in Democratic caucus; all four modeled independent opposition contenders unassigned, matching stated campaign pledges without treating pledges as certain future conduct.
          - generic [ref=e219]:
            - generic [ref=e220]:
              - term [ref=e221]: Democratic seats
              - definition [ref=e222]: 50.2 average · 46–55 in the middle 90% of simulated elections
            - generic [ref=e223]:
              - term [ref=e224]: Republican seats
              - definition [ref=e225]: 49.0 average · 44–54 in the middle 90% of simulated elections
            - generic [ref=e226]:
              - term [ref=e227]: Other / unassigned seats
              - definition [ref=e228]: 0.7 average · 0–2 in the middle 90% of simulated elections
          - paragraph [ref=e229]: These chamber ranges come from the saved election simulations. Races can miss in the same direction together; they are not independent tests of a national prediction.
        - paragraph [ref=e230]:
          - strong [ref=e231]: 50/50 does not tell us the margin.
          - text: It can mean a narrow contest, or uncertainty about which candidate wins comfortably. Use the separate margin range and close-race chances where available.
      - region [ref=e232]:
        - generic [ref=e233]: The evaluation rubric
        - heading "How the forecast will be judged" [level=3] [ref=e234]
        - generic "A through F scale; no grade assigned" [ref=e235]:
          - generic [ref=e236]: A
          - generic [ref=e237]: B
          - generic [ref=e238]: C
          - generic [ref=e239]: D
          - generic [ref=e240]: F
        - paragraph [ref=e241]:
          - strong [ref=e242]: No letter grade assigned.
          - text: The thresholds, comparison forecasts and rules for combining scores are still under review. They must be published before outcomes are scored.
        - generic [ref=e243]:
          - generic [ref=e244]:
            - term [ref=e245]: Specific predictions
            - definition [ref=e246]: Compare the named candidate margins with verified final results, using the same vote definition. Show the size and direction of each error.
          - generic [ref=e247]:
            - term [ref=e248]: Useful probabilities
            - definition [ref=e249]: Score confident mistakes more heavily and compare performance with simple forecasts on the same races. Predicting everything at 50/50 must not pass as demonstrated skill.
          - generic [ref=e250]:
            - term [ref=e251]: Honest ranges
            - definition [ref=e252]: Check both how often results fall inside the ranges and how wide those ranges were. Very broad ranges do not earn a high grade merely by containing the results.
          - generic [ref=e253]:
            - term [ref=e254]: Shared misses
            - definition [ref=e255]: Check coordinated shifts against the model’s joint predictions. Many misses in one election are not many independent elections.
        - paragraph [ref=e256]: A result having been possible does not excuse poor performance. One surprising result also does not settle whether the model is reliable across elections. Both the individual misses and the comparison scores need to stay visible.
      - group [ref=e257]:
        - generic "Edition, sources and evaluation definitions" [ref=e258] [cursor=pointer]
      - button "Return to the forecast" [ref=e259] [cursor=pointer]
    - tabpanel "Senate" [ref=e260]:
      - main [ref=e261]:
        - generic [ref=e263]:
          - generic [ref=e264]: 2026 United States Senate · 35 elections
          - heading "The Senate, in perspective." [level=1] [ref=e265]: The Senate,in perspective.
          - paragraph [ref=e266]: Choose the model’s ingredients. Explore the races and chamber together.A closer look at the races that could shape the chamber, with visible uncertainty.
        - group [ref=e267]:
          - generic "Build your perspective Electoral history Expand to customize" [ref=e268] [cursor=pointer]:
            - generic [ref=e269]:
              - generic [ref=e270]: Build your perspective
              - generic [ref=e271]: Electoral history
            - generic [ref=e272]:
              - generic [ref=e273]: Expand to customize
              - generic [aria-hidden] [ref=e274]: ⌄
          - option "October 20, 2026" [selected]
          - option "Electoral-history starting estimate" [selected]
          - option "Recent calendar history"
          - option "Refreshed history"
          - option "Learned presidential slope"
          - option "Relative lean with presidential context"
          - option "Relative state and national lean"
          - option "Standard uncertainty" [selected]
          - option "Polling-error sizes × 2/3"
          - option "Polling-error sizes × 1.5"
          - option "Shared election-wide error size × 0.5"
          - option "Shared election-wide error size × 2"
          - option "2026 default · Different forecast definition" [disabled] [selected]
          - option "Electoral history"
        - generic [ref=e275]:
          - generic [ref=e276]:
            - generic [ref=e277]: Electoral history · 100,000 saved simulations
            - generic [ref=e278]:
              - generic [ref=e279]: Who joins each party?
              - combobox "Who joins each party?" [ref=e280]:
                - option "Independents unaligned" [selected]
                - option "Independents work with Democrats"
                - option "Independents work with Republicans"
          - paragraph [ref=e281]: Each simulation is one possible set of election results. A caucus is a party’s group in the Senate, including independents who work with it. Choose those alliances above; candidate party labels stay the same.
          - region [ref=e282]:
            - generic [ref=e283]:
              - generic [ref=e284]:
                - heading "The path to a majority" [level=2] [ref=e285]
                - paragraph [ref=e286]: 35 elections, ordered by the projected vote-share lead between the two modeled contenders.
              - generic [ref=e287]:
                - generic [ref=e288]: D
                - generic [ref=e289]: R
                - text: Projected control thresholds
            - generic [aria-hidden] [ref=e290]:
              - generic [ref=e291]: Democratic / independent advantage
              - generic [ref=e292]: Republican advantage
            - region "Senate race ribbon, ordered by projected margin. Scroll horizontally for all 35 elections." [ref=e293]:
              - generic [ref=e294]:
                - generic [ref=e295]:
                  - strong [ref=e296]: "34"
                  - generic [ref=e297]: D caucusnot up this year
                - button "Massachusetts, regular election, Safe, projected margin D +32.6 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e298] [cursor=pointer]:
                  - generic [ref=e299]: MA
                  - generic [ref=e300]: ’26
                - button "Delaware, regular election, Safe, projected margin D +22.3 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations." [ref=e301] [cursor=pointer]:
                  - generic [ref=e302]: DE
                  - generic [ref=e303]: ’26
                - button "Oregon, regular election, Safe, projected margin D +22.2 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations." [ref=e304] [cursor=pointer]:
                  - generic [ref=e305]: OR
                  - generic [ref=e306]: ’26
                - button "Rhode Island, regular election, Safe, projected margin D +21.4 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.3% of simulations." [ref=e307] [cursor=pointer]:
                  - generic [ref=e308]: RI
                  - generic [ref=e309]: ’26
                - button "Colorado, regular election, Safe, projected margin D +18.7 pp. At the D control threshold in 0.4% of simulations. At the R control threshold in 0.4% of simulations." [ref=e310] [cursor=pointer]:
                  - generic [ref=e311]: CO
                  - generic [ref=e312]: ’26
                - button "New Mexico, regular election, Likely, projected margin D +13.7 pp. At the D control threshold in 0.8% of simulations. At the R control threshold in 0.8% of simulations." [ref=e313] [cursor=pointer]:
                  - generic [ref=e314]: NM
                  - generic [ref=e315]: ’26
                - button "New Jersey, regular election, Likely, projected margin D +13.7 pp. At the D control threshold in 0.8% of simulations. At the R control threshold in 0.9% of simulations." [ref=e316] [cursor=pointer]:
                  - generic [ref=e317]: NJ
                  - generic [ref=e318]: ’26
                - button "Virginia, regular election, Likely, projected margin D +13.5 pp. At the D control threshold in 0.9% of simulations. At the R control threshold in 0.9% of simulations." [ref=e319] [cursor=pointer]:
                  - generic [ref=e320]: VA
                  - generic [ref=e321]: ’26
                - button "Illinois, regular election, Likely, projected margin D +13.1 pp. At the D control threshold in 0.9% of simulations. At the R control threshold in 0.9% of simulations." [ref=e322] [cursor=pointer]:
                  - generic [ref=e323]: IL
                  - generic [ref=e324]: ’26
                - button "Minnesota, regular election, Likely, projected margin D +6.5 pp. At the D control threshold in 2.5% of simulations. At the R control threshold in 2.6% of simulations." [ref=e325] [cursor=pointer]:
                  - generic [ref=e326]: MN
                  - generic [ref=e327]: ’26
                - button "Georgia, regular election, Likely, projected margin D +5.6 pp. At the D control threshold in 2.9% of simulations. At the R control threshold in 2.9% of simulations." [ref=e328] [cursor=pointer]:
                  - generic [ref=e329]: GA
                  - generic [ref=e330]: ’26
                - button "New Hampshire, regular election, Likely, projected margin D +5.0 pp. At the D control threshold in 3.1% of simulations. At the R control threshold in 3.1% of simulations." [ref=e331] [cursor=pointer]:
                  - generic [ref=e332]: NH
                  - generic [ref=e333]: ’26
                - button "Maine, regular election, Likely, projected margin D +3.6 pp. At the D control threshold in 3.7% of simulations. At the R control threshold in 3.8% of simulations." [ref=e334] [cursor=pointer]:
                  - generic [ref=e335]: ME
                  - generic [ref=e336]: ’26
                - button "Michigan, regular election, Tossup, projected margin D +0.8 pp. At the D control threshold in 5.4% of simulations. At the R control threshold in 5.4% of simulations." [pressed] [ref=e337] [cursor=pointer]:
                  - generic [ref=e338]: MI
                  - generic [ref=e339]: ’26
                - button "North Carolina, regular election, Tossup, projected margin R +0.9 pp. At the D control threshold in 6.6% of simulations. At the R control threshold in 6.6% of simulations." [ref=e340] [cursor=pointer]:
                  - generic [ref=e341]: NC
                  - generic [ref=e342]: ’26
                - button "Iowa, regular election, Likely, projected margin R +11.0 pp. At the D control threshold in 11.2% of simulations. At the R control threshold in 10.1% of simulations." [ref=e343] [cursor=pointer]:
                  - generic [ref=e344]: IA
                  - generic [ref=e345]: ’26
                - button "Texas, regular election, Likely, projected margin R +11.5 pp, projected D control threshold, projected R control threshold. At the D control threshold in 11.1% of simulations. At the R control threshold in 10.0% of simulations." [ref=e346] [cursor=pointer]:
                  - generic [ref=e348]: D51 / R50 · TX
                  - generic [ref=e349]: TX
                  - generic [ref=e350]: ’26
                - button "Ohio, special election, Likely, projected margin R +14.4 pp. At the D control threshold in 9.6% of simulations. At the R control threshold in 8.4% of simulations." [ref=e351] [cursor=pointer]:
                  - generic [ref=e352]: OH
                  - generic [ref=e353]: SP
                - button "Florida, special election, Safe, projected margin R +16.3 pp. At the D control threshold in 8.1% of simulations. At the R control threshold in 7.0% of simulations." [ref=e354] [cursor=pointer]:
                  - generic [ref=e355]: FL
                  - generic [ref=e356]: SP
                - button "Alaska, regular election, Safe, projected margin R +16.8 pp. At the D control threshold in 8.0% of simulations. At the R control threshold in 6.8% of simulations." [ref=e357] [cursor=pointer]:
                  - generic [ref=e358]: AK
                  - generic [ref=e359]: ’26
                - button "Montana, regular election, Safe, projected margin R +18.0 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 5.8% of simulations." [ref=e360] [cursor=pointer]:
                  - generic [ref=e361]: MT
                  - generic [ref=e362]: ’26
                - button "Kansas, regular election, Safe, projected margin R +19.4 pp. At the D control threshold in 5.7% of simulations. At the R control threshold in 4.9% of simulations." [ref=e363] [cursor=pointer]:
                  - generic [ref=e364]: KS
                  - generic [ref=e365]: ’26
                - button "Louisiana, regular election, Safe, projected margin R +19.9 pp. At the D control threshold in 5.4% of simulations. At the R control threshold in 4.6% of simulations." [ref=e366] [cursor=pointer]:
                  - generic [ref=e367]: LA
                  - generic [ref=e368]: ’26
                - button "South Carolina, regular election, Safe, projected margin R +21.1 pp. At the D control threshold in 4.5% of simulations. At the R control threshold in 3.8% of simulations." [ref=e369] [cursor=pointer]:
                  - generic [ref=e370]: SC
                  - generic [ref=e371]: ’26
                - button "Nebraska, regular election, Safe, projected margin R +23.7 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.6% of simulations." [ref=e372] [cursor=pointer]:
                  - generic [ref=e373]: NE
                  - generic [ref=e374]: ’26
                - button "Mississippi, regular election, Safe, projected margin R +25.9 pp. At the D control threshold in 2.2% of simulations. At the R control threshold in 1.8% of simulations." [ref=e375] [cursor=pointer]:
                  - generic [ref=e376]: MS
                  - generic [ref=e377]: ’26
                - button "Alabama, regular election, Safe, projected margin R +28.4 pp. At the D control threshold in 1.5% of simulations. At the R control threshold in 1.2% of simulations." [ref=e378] [cursor=pointer]:
                  - generic [ref=e379]: AL
                  - generic [ref=e380]: ’26
                - button "Kentucky, regular election, Safe, projected margin R +28.6 pp. At the D control threshold in 1.5% of simulations. At the R control threshold in 1.2% of simulations." [ref=e381] [cursor=pointer]:
                  - generic [ref=e382]: KY
                  - generic [ref=e383]: ’26
                - button "Oklahoma, regular election, Safe, projected margin R +32.4 pp. At the D control threshold in 0.8% of simulations. At the R control threshold in 0.6% of simulations." [ref=e384] [cursor=pointer]:
                  - generic [ref=e385]: OK
                  - generic [ref=e386]: ’26
                - button "South Dakota, regular election, Safe, projected margin R +32.5 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.6% of simulations." [ref=e387] [cursor=pointer]:
                  - generic [ref=e388]: SD
                  - generic [ref=e389]: ’26
                - button "Tennessee, regular election, Safe, projected margin R +32.7 pp. At the D control threshold in 0.7% of simulations. At the R control threshold in 0.6% of simulations." [ref=e390] [cursor=pointer]:
                  - generic [ref=e391]: TN
                  - generic [ref=e392]: ’26
                - button "Arkansas, regular election, Safe, projected margin R +33.9 pp. At the D control threshold in 0.6% of simulations. At the R control threshold in 0.5% of simulations." [ref=e393] [cursor=pointer]:
                  - generic [ref=e394]: AR
                  - generic [ref=e395]: ’26
                - button "Idaho, regular election, Safe, projected margin R +39.9 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.2% of simulations." [ref=e396] [cursor=pointer]:
                  - generic [ref=e397]: ID
                  - generic [ref=e398]: ’26
                - button "Wyoming, regular election, Safe, projected margin R +44.5 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e399] [cursor=pointer]:
                  - generic [ref=e400]: WY
                  - generic [ref=e401]: ’26
                - button "West Virginia, regular election, Safe, projected margin R +44.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e402] [cursor=pointer]:
                  - generic [ref=e403]: WV
                  - generic [ref=e404]: ’26
                - generic [ref=e405]:
                  - strong [ref=e406]: "31"
                  - generic [ref=e407]: R caucusnot up this year
            - paragraph [ref=e408]: Read inward from either party’s end. Its marker identifies the last race needed for control in this projected order, after seats not up for election and races guaranteed to join that party’s group. D = Democratic; R = Republican. “SP” marks a special election to fill an unfinished term.
            - generic [ref=e409]:
              - generic [ref=e410]:
                - generic [ref=e411]: D
                - generic [ref=e412]:
                  - strong [ref=e413]:
                    - button "Texas" [ref=e414] [cursor=pointer]
                  - generic [ref=e415]: 51 caucus seats for control · projected margin −11.5 pp
                  - generic [ref=e416]:
                    - text: "Most frequent in simulations:"
                    - button "Iowa" [ref=e417] [cursor=pointer]
                    - text: · 11.2%
              - generic [ref=e418]:
                - generic [ref=e419]: R
                - generic [ref=e420]:
                  - strong [ref=e421]:
                    - button "Texas" [ref=e422] [cursor=pointer]
                  - generic [ref=e423]: 50 caucus seats for control · projected margin +11.5 pp
                  - generic [ref=e424]:
                    - text: "Most frequent in simulations:"
                    - button "Iowa" [ref=e425] [cursor=pointer]
                    - text: · 10.1%
            - generic [ref=e426]:
              - strong [ref=e427]: Michigan · D +0.8 pp
              - generic [ref=e428]: At the D control threshold in 5.4% of simulations.
              - generic [ref=e429]: At the R control threshold in 5.4% of simulations.
            - paragraph [ref=e430]: "A margin is the difference in vote shares: 52% versus 48% is a 4 percentage-point (pp) lead. The frequencies above count how often a race occupies the control-threshold rank across simulated elections, including simulations where that party falls short. They are not the chance that flipping that race alone would change control."
            - group [ref=e431]:
              - generic "How to read the thresholds" [ref=e432] [cursor=pointer]
        - region "Senate race explorer" [ref=e433]:
          - generic [ref=e434]:
            - generic [ref=e435]:
              - generic [ref=e436]:
                - generic [ref=e437]:
                  - heading "A nation of distinct races" [level=2] [ref=e438]
                  - generic [ref=e439]: Select a state to explore its election.
                - generic [ref=e440]: 2026 Senate
              - group "United States Senate races. Use Tab to select a state with an election." [ref=e441]:
                - button "New Mexico, 1 election, Likely" [ref=e442] [cursor=pointer]
                - button "South Dakota, 1 election, Safe" [ref=e443] [cursor=pointer]
                - generic "California, no Senate election" [ref=e444]
                - button "Kentucky, 1 election, Safe" [ref=e445] [cursor=pointer]
                - button "Alabama, 1 election, Safe" [ref=e446] [cursor=pointer]
                - button "Georgia, 1 election, Likely" [ref=e447] [cursor=pointer]
                - button "Arkansas, 1 election, Safe" [ref=e448] [cursor=pointer]
                - generic "Pennsylvania, no Senate election" [ref=e449]
                - generic "Missouri, no Senate election" [ref=e450]
                - button "Colorado, 1 election, Safe" [ref=e451] [cursor=pointer]
                - generic "Utah, no Senate election" [ref=e452]
                - button "Oklahoma, 1 election, Safe" [ref=e453] [cursor=pointer]
                - button "Tennessee, 1 election, Safe" [ref=e454] [cursor=pointer]
                - button "Wyoming, 1 election, Safe" [ref=e455] [cursor=pointer]
                - generic "New York, no Senate election" [ref=e456]
                - generic "Indiana, no Senate election" [ref=e457]
                - button "Kansas, 1 election, Safe" [ref=e458] [cursor=pointer]
                - button "Idaho, 1 election, Safe" [ref=e459] [cursor=pointer]
                - button "Alaska, 1 election, Safe" [ref=e460] [cursor=pointer]
                - generic "Nevada, no Senate election" [ref=e461]
                - button "Illinois, 1 election, Likely" [ref=e462] [cursor=pointer]
                - generic "Vermont, no Senate election" [ref=e463]
                - button "New Jersey, 1 election, Likely" [ref=e464] [cursor=pointer]
                - generic "North Dakota, no Senate election" [ref=e465]
                - button "Iowa, 1 election, Likely" [ref=e466] [cursor=pointer]
                - button "South Carolina, 1 election, Safe" [ref=e467] [cursor=pointer]
                - generic "Arizona, no Senate election" [ref=e468]
                - button "Delaware, 1 election, Safe" [ref=e469] [cursor=pointer]
                - generic "District of Columbia, no Senate election" [ref=e470]
                - generic "Connecticut, no Senate election" [ref=e471]
                - button "New Hampshire, 1 election, Likely" [ref=e472] [cursor=pointer]
                - button "Nebraska, 1 election, Safe" [ref=e473] [cursor=pointer]
                - generic "Washington, no Senate election" [ref=e474]
                - button "Texas, 1 election, Likely" [ref=e475] [cursor=pointer]
                - button "Ohio, 1 election, Likely" [ref=e476] [cursor=pointer]
                - generic "Wisconsin, no Senate election" [ref=e477]
                - button "Oregon, 1 election, Safe" [ref=e478] [cursor=pointer]
                - button "Mississippi, 1 election, Safe" [ref=e479] [cursor=pointer]
                - button "North Carolina, 1 election, Tossup" [ref=e480] [cursor=pointer]
                - button "Virginia, 1 election, Likely" [ref=e481] [cursor=pointer]
                - button "West Virginia, 1 election, Safe" [ref=e482] [cursor=pointer]
                - button "Louisiana, 1 election, Safe" [ref=e483] [cursor=pointer]
                - button "Michigan, 1 election, Tossup" [pressed] [ref=e484] [cursor=pointer]
                - button "Massachusetts, 1 election, Safe" [ref=e485] [cursor=pointer]
                - button "Florida, 1 election, Safe" [ref=e486] [cursor=pointer]
                - button "Montana, 1 election, Safe" [ref=e487] [cursor=pointer]
                - button "Minnesota, 1 election, Likely" [ref=e488] [cursor=pointer]
                - generic "Maryland, no Senate election" [ref=e489]
                - button "Maine, 1 election, Likely" [ref=e490] [cursor=pointer]
                - generic "Hawaii, no Senate election" [ref=e491]
                - button "Rhode Island, 1 election, Safe" [ref=e492] [cursor=pointer]
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
                - generic [ref=e493]: Alaska and Hawaii shown in insets; scale varies.
              - generic [ref=e494]:
                - generic [ref=e495]: Safe D
                - generic [ref=e497]: Likely D
                - generic [ref=e499]: Safe R
                - generic [ref=e501]: Likely R
                - generic [ref=e503]: Safe I
                - generic [ref=e505]: Likely I
                - generic [ref=e507]: Tossup
              - generic [ref=e509]:
                - generic [ref=e510]: "Ratings use the favorite’s chance of winning: Safe above 90%; Likely above 60% through 90%; Tossup 60% or lower (40–60% each in a two-outcome race). Safe does not mean certain."
                - generic [ref=e511]: "D = Democratic · R = Republican · I = Independent. Gray: no election. State area does not represent votes."
            - region [ref=e512]:
              - generic [ref=e514]:
                - heading "Who could control the chamber?" [level=2] [ref=e515]
                - generic [ref=e516]: Independents unaligned · D needs 51; R needs 50
              - generic [ref=e517]:
                - generic [ref=e518]:
                  - generic [ref=e519]: Democratic control
                  - generic [ref=e521]: 9.4%
                - generic [ref=e522]:
                  - generic [ref=e523]: Republican control
                  - generic [ref=e525]: 89.4%
                - generic [ref=e526]:
                  - generic [ref=e527]: Control depends on alliances
                  - generic [ref=e529]: 1.3%
              - paragraph [ref=e534]: King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
              - paragraph [ref=e535]: A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.
              - generic [ref=e536]:
                - generic [ref=e537]:
                  - text: "D caucus: 46.8 expected seats"
                  - generic [ref=e538]: (42–52, modeled 90% range)
                - generic [ref=e539]:
                  - text: "R caucus: 53.0 expected seats"
                  - generic [ref=e540]: (48–58, modeled 90% range)
                - generic [ref=e541]:
                  - text: "Unaligned: 0.1 expected seats"
                  - generic [ref=e542]: (0–1, modeled 90% range)
              - paragraph [ref=e543]: Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.
            - region [ref=e544]:
              - heading "What has been tested?" [level=2] [ref=e545]
              - paragraph [ref=e546]: The 2026 election has not happened yet. Select 2018 or 2022 to see how historical reconstructions compare with actual results. Those views use a different model of possible winners, so their scores do not measure the accuracy of this 2026 two-contender forecast.
              - group [ref=e547]:
                - generic "Scope and limitations" [ref=e548] [cursor=pointer]
            - region [ref=e549]:
              - generic [ref=e550]:
                - heading "The races" [level=2] [ref=e551]
                - generic [ref=e552]: 10 of 35 elections
              - paragraph [ref=e553]: The favorite is the candidate or group with the highest modeled chance of winning. “Most competitive” puts the least certain favorites first.
              - generic [ref=e554]:
                - generic [ref=e555]:
                  - generic [aria-hidden] [ref=e556]: ⌕
                  - textbox "Search states or candidates" [ref=e557]:
                    - /placeholder: Find a state or candidate
                - combobox "Sort races" [ref=e558]:
                  - option "Most competitive" [selected]
                  - option "State A–Z"
              - table [ref=e559]:
                - rowgroup [ref=e560]:
                  - row [ref=e561]:
                    - columnheader "State / election" [ref=e562]
                    - columnheader "Favorite" [ref=e563]
                    - columnheader "Chance to win" [ref=e564]
                - rowgroup [ref=e565]:
                  - row [ref=e566]:
                    - cell [ref=e567]:
                      - button "Michigan Regular election" [pressed] [ref=e568] [cursor=pointer]:
                        - text: Michigan
                        - generic [ref=e569]: Regular election
                    - cell "Abdul El-Sayed Democratic" [ref=e570]:
                      - text: Abdul El-Sayed
                      - generic [ref=e571]: Democratic
                    - cell "52.8%" [ref=e572]
                  - row [ref=e576]:
                    - cell [ref=e577]:
                      - button "North Carolina Regular election" [ref=e578] [cursor=pointer]:
                        - text: North Carolina
                        - generic [ref=e579]: Regular election
                    - cell "Michael Whatley Republican" [ref=e580]:
                      - text: Michael Whatley
                      - generic [ref=e581]: Republican
                    - cell "53.3%" [ref=e582]
                  - row [ref=e586]:
                    - cell [ref=e587]:
                      - button "Maine Regular election" [ref=e588] [cursor=pointer]:
                        - text: Maine
                        - generic [ref=e589]: Regular election
                    - cell "Jackson, Troy D. Democratic" [ref=e590]:
                      - text: Jackson, Troy D.
                      - generic [ref=e591]: Democratic
                    - cell "62.0%" [ref=e592]
                  - row [ref=e596]:
                    - cell [ref=e597]:
                      - button "New Hampshire Regular election" [ref=e598] [cursor=pointer]:
                        - text: New Hampshire
                        - generic [ref=e599]: Regular election
                    - cell "Chris Pappas Democratic" [ref=e600]:
                      - text: Chris Pappas
                      - generic [ref=e601]: Democratic
                    - cell "66.7%" [ref=e602]
                  - row [ref=e606]:
                    - cell [ref=e607]:
                      - button "Georgia Regular election" [ref=e608] [cursor=pointer]:
                        - text: Georgia
                        - generic [ref=e609]: Regular election
                    - cell "Jon Ossoff Democratic" [ref=e610]:
                      - text: Jon Ossoff
                      - generic [ref=e611]: Democratic
                    - cell "68.5%" [ref=e612]
                  - row [ref=e616]:
                    - cell [ref=e617]:
                      - button "Minnesota Regular election" [ref=e618] [cursor=pointer]:
                        - text: Minnesota
                        - generic [ref=e619]: Regular election
                    - cell "Peggy Flanagan Democratic" [ref=e620]:
                      - text: Peggy Flanagan
                      - generic [ref=e621]: Democratic
                    - cell "71.3%" [ref=e622]
                  - row [ref=e626]:
                    - cell [ref=e627]:
                      - button "Iowa Regular election" [ref=e628] [cursor=pointer]:
                        - text: Iowa
                        - generic [ref=e629]: Regular election
                    - cell "Ashley Hinson Republican" [ref=e630]:
                      - text: Ashley Hinson
                      - generic [ref=e631]: Republican
                    - cell "82.3%" [ref=e632]
                  - row [ref=e636]:
                    - cell [ref=e637]:
                      - button "Texas Regular election" [ref=e638] [cursor=pointer]:
                        - text: Texas
                        - generic [ref=e639]: Regular election
                    - cell "KEN PAXTON Republican" [ref=e640]:
                      - text: KEN PAXTON
                      - generic [ref=e641]: Republican
                    - cell "83.2%" [ref=e642]
                  - row [ref=e646]:
                    - cell [ref=e647]:
                      - button "Illinois Regular election" [ref=e648] [cursor=pointer]:
                        - text: Illinois
                        - generic [ref=e649]: Regular election
                    - cell "Juliana Stratton Democratic" [ref=e650]:
                      - text: Juliana Stratton
                      - generic [ref=e651]: Democratic
                    - cell "86.0%" [ref=e652]
                  - row [ref=e656]:
                    - cell [ref=e657]:
                      - button "Virginia Regular election" [ref=e658] [cursor=pointer]:
                        - text: Virginia
                        - generic [ref=e659]: Regular election
                    - cell "Mark R. Warner Democratic" [ref=e660]:
                      - text: Mark R. Warner
                      - generic [ref=e661]: Democratic
                    - cell "86.5%" [ref=e662]
              - button "See all 35 races" [ref=e666] [cursor=pointer]:
                - text: See all 35 races
                - generic [aria-hidden] [ref=e667]: ⌄
          - complementary "Selected race detail" [ref=e668]:
            - generic [ref=e669]:
              - generic [ref=e670]: Race in focus
              - generic [ref=e671]: Tossup
            - heading "Michigan" [level=2] [ref=e672]
            - generic [ref=e673]: U.S. Senate · Regular election
            - separator [ref=e674]
            - heading "Chance of winning" [level=3] [ref=e675]
            - paragraph [ref=e676]: A 70% chance means winning about 70 out of 100 elections simulated by this model. It does not mean winning 70% of the votes.
            - generic [ref=e677]:
              - generic [ref=e679]:
                - generic [ref=e680]:
                  - generic [ref=e681]: Abdul El-Sayed
                  - generic [ref=e682]: Democratic
                - strong [ref=e683]: 52.8%
              - generic [ref=e687]:
                - generic [ref=e688]:
                  - generic [ref=e689]: Mike Rogers
                  - generic [ref=e690]: Republican
                - strong [ref=e691]: 47.2%
            - separator [ref=e694]
            - heading "Vote share among these two contenders" [level=3] [ref=e695]
            - generic [ref=e696]:
              - generic [ref=e697]:
                - generic [ref=e698]:
                  - generic [ref=e699]: El-Sayed
                  - strong [ref=e700]: 50.4%
                - 'generic "Abdul El-Sayed: mean 50.4%, Modeled 90% range 39.6% to 61.1%" [ref=e701]'
                - paragraph [ref=e704]: "Modeled 90% range: 39.6%–61.1%Share of votes for the two named principal contenders"
              - generic [ref=e705]:
                - generic [ref=e706]:
                  - generic [ref=e707]: Rogers
                  - strong [ref=e708]: 49.6%
                - 'generic "Mike Rogers: mean 49.6%, Modeled 90% range 38.9% to 60.4%" [ref=e709]'
                - paragraph [ref=e712]: "Modeled 90% range: 38.9%–60.4%Share of votes for the two named principal contenders"
            - paragraph [ref=e713]: The shares divide votes between these two contenders only; votes for other candidates are excluded. The range runs from the 5th to the 95th percentile of simulated shares—the middle 90%. It is a model estimate, not a poll’s margin of error or a guarantee.
            - paragraph [ref=e714]: Probabilities of 0% or 100% reflect the saved simulation, not certainty.
            - separator [ref=e715]
            - generic [ref=e717]:
              - text: Selected model setup
              - strong [ref=e718]: Electoral history
            - paragraph [ref=e719]: This setup uses electoral history without polls.
            - group [ref=e720]:
              - generic "Race assumptions" [ref=e721] [cursor=pointer]
            - separator [ref=e722]
            - heading "Candidate and result sources" [level=3] [ref=e723]
            - generic [ref=e724]:
              - link "Abdul El-Sayed ↗" [ref=e726] [cursor=pointer]:
                - /url: https://www.isabellacounty.org/departments/county-clerk/services/general-election-information/
              - link "Mike Rogers ↗" [ref=e728] [cursor=pointer]:
                - /url: https://www.isabellacounty.org/departments/county-clerk/services/general-election-information/
            - paragraph [ref=e729]: "Forecast date: October 20, 2026Inputs dated through: October 20, 2026Saved research forecast; new polls are not added automatically."
        - region [ref=e730]:
          - generic [ref=e731]:
            - generic [ref=e732]: From evidence to possible elections
            - heading "How this forecast works." [level=2] [ref=e733]
            - paragraph [ref=e734]: A model is a set of mathematical assumptions connecting evidence to possible results. Your choices change those assumptions and, when available, load the matching saved forecast.
            - link "Download this forecast’s data ↓" [ref=e735] [cursor=pointer]:
              - /url: /data/explorer/current-fundamentals.json
            - paragraph [ref=e736]: "The download is a JSON file: structured data for analysis, with source identifiers and limitations."
          - generic [ref=e737]:
            - generic [ref=e738]:
              - heading "1. Start with a baseline" [level=3] [ref=e739]
              - paragraph [ref=e740]: Electoral history supplies a starting estimate, often called “fundamentals.” The builder also offers a historical winner reference model. A component is one part of the model; the chosen combination is your model setup or “stack.”
            - generic [ref=e741]:
              - heading "2. Add the selected evidence" [level=3] [ref=e742]
              - paragraph [ref=e743]:
                - text: This setup uses no polls. Polls measure responses at a particular time. The model allows for measurement errors and changes before election day.
                - link "Read the poll selection rules below." [ref=e744] [cursor=pointer]:
                  - /url: "#poll-selection"
            - generic [ref=e745]:
              - heading "3. Simulate elections together" [level=3] [ref=e746]
              - paragraph [ref=e747]: A simulation is one possible set of election results generated by the model. Shared uncertainty allows many races to move together—for example, if polls miss in the same direction. Win chances and chamber control summarize these simulated elections, rather than assuming each race is unrelated.
            - generic [ref=e748]:
              - heading "4. Explore what changes" [level=3] [ref=e749]
              - paragraph [ref=e750]: Changing a choice loads a saved result if one exists. Some combinations require calculating several model parts together again (a “joint refit”); the website explains when that work is needed. A sensitivity test changes assumptions without adding evidence. “Runs successfully,” “experimental,” and “tested against election results” describe different things.
            - generic [ref=e751]:
              - heading "What these numbers cover" [level=3] [ref=e752]
              - paragraph [ref=e753]: The 2026 preview models only the two named contenders in each race. Its vote shares divide votes between those two; other candidates’ shares and chances of winning are outside this model.
            - generic [ref=e754]:
              - heading "Make fair comparisons" [level=3] [ref=e755]
              - paragraph [ref=e756]: Compare the same year, possible winners and assumptions about independent senators. A change from 40% to 45% is 5 percentage points (“pp”), not a 5% relative increase. Historical scores describe those elections and do not establish the accuracy of the current forecast.
        - region [ref=e757]:
          - generic [ref=e758]: The evidence behind this view
          - heading "Why some polls are included—and others are not" [level=2] [ref=e759]
          - paragraph [ref=e760]: This model setup uses electoral history without polling. No polls are included. Choose a polling component in “Build your perspective” to explore a saved forecast that uses polls.
        - generic [ref=e761]:
          - generic [ref=e762]: ballotmancy
          - generic [ref=e763]: Independent research. Visible assumptions.2026 Senate research preview · October 20, 2026
  - alert [ref=e764]
```

# Test source

```ts
  1  | import { test, expect } from "@playwright/test";
  2  | import { readFileSync } from "node:fs";
  3  | import { resolve } from "node:path";
  4  | 
  5  | const read = (path: string) => JSON.parse(readFileSync(resolve(path), "utf8"));
  6  | const senateRegistry = read("public/data/explorer/registry.json");
  7  | const houseCatalog = read("data/house-live/catalog.json");
  8  | 
  9  | for (const chamber of ["senate", "house"] as const) {
  10 |   test(`${chamber}: same-page guide starts standard, renders every exact model, and preserves the visitor's forecast`, async ({ page }, testInfo) => {
  11 |     test.setTimeout(240_000);
  12 |     const errors: string[] = [];
  13 |     page.on("pageerror", error => errors.push(error.message));
  14 |     const initialRun = chamber === "senate" ? "current-fundamentals" : "house-current-fundamentals";
  15 |     const defaultId = chamber === "senate" ? senateRegistry.defaults["2026"] : houseCatalog.default_id;
  16 |     const entries = chamber === "senate" ? senateRegistry.runs.filter((entry: { year: number }) => entry.year === 2026) : houseCatalog.entries;
  17 |     await page.goto(chamber === "senate" ? `/?run=${initialRun}` : `/?chamber=house&edition=current&run=${initialRun}`);
  18 |     await expect(page.locator(".hero")).toBeVisible();
  19 |     const originalUrl = page.url();
  20 |     const originalHero = await page.locator(".hero").textContent();
  21 |     const trigger = page.getByRole("button", { name: "Election night scorecard", exact: true });
  22 |     await trigger.click();
  23 |     const panel = page.locator("#election-night-scorecard");
  24 |     await expect(panel.getByRole("heading", { name: "Election night scorecard", exact: true })).toBeFocused();
  25 |     await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
  26 |     await expect(panel).toHaveAttribute("aria-busy", "false");
  27 |     await expect(panel.locator(".night-featured .night-race")).toHaveCount(6);
  28 |     const watchIds = await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")));
  29 |     await panel.locator(".night-models summary").click();
  30 |     const select = panel.getByLabel("Scorecard model");
  31 |     await expect(select).toHaveValue(defaultId);
  32 |     await expect(select.locator("option")).toHaveCount(16);
  33 |     for (const entry of entries) {
  34 |       const raw = chamber === "senate" ? read(`public${entry.href}`) : read(`data/house-live/${entry.result.path}`);
  35 |       const digest = chamber === "senate" ? entry.sha256 : entry.result.sha256;
  36 |       const missed = raw.races.reduce((sum: number, race: { outcomes: { winProbability?: number; p_win?: number }[] }) => sum +
  37 |         1 - Math.max(...race.outcomes.map(outcome => chamber === "senate" ? outcome.winProbability! : outcome.p_win!)), 0);
  38 |       await select.selectOption(entry.id);
  39 |       await expect(panel.locator(".night-method code")).toHaveText(digest);
  40 |       await expect(panel.locator(".night-edition strong")).toContainText(entry.label);
  41 |       await expect(panel.locator(".night-big-number")).toContainText(missed.toFixed(1));
  42 |       expect(await panel.locator(".night-featured .night-race").evaluateAll(nodes => nodes.map(node => node.getAttribute("data-scorecard-race")))).toEqual(watchIds);
  43 |       expect(page.url()).toBe(originalUrl);
  44 |       expect(await page.locator(".hero").textContent()).toBe(originalHero);
  45 |     }
  46 |     await expect(panel.locator(".night-rubric")).toContainText("No letter grade assigned");
  47 |     await expect(panel.locator(".night-edition")).toContainText("have not yet been locked");
  48 |     await expect(panel.locator(".night-uncertainty-grid")).toContainText("not a limit or a measured accuracy record");
  49 |     await panel.locator(".night-featured").screenshot({ path: testInfo.outputPath(`${chamber}-featured.png`) });
  50 |     await panel.locator(".night-uncertainty-grid").screenshot({ path: testInfo.outputPath(`${chamber}-uncertainty.png`) });
> 51 |     await panel.locator(".night-all-races summary").click();
     |                                                     ^ Error: locator.click: Error: strict mode violation: locator('#election-night-scorecard').locator('.night-all-races summary') resolved to 11 elements:
  52 |     await expect(panel.locator(".night-predictions .night-race")).toHaveCount(10);
  53 |     if (chamber === "house") {
  54 |       const defaultEntry = entries.find((entry: { id: string }) => entry.id === defaultId);
  55 |       const unsupported = read(`data/house-live/${defaultEntry.result.path}`).races.find((race: { conditional_shares: unknown }) => race.conditional_shares === null);
  56 |       await panel.getByLabel("Find a race or candidate").fill(unsupported.district_key);
  57 |       await expect(panel.locator(".night-predictions .night-race")).toHaveCount(1);
  58 |       await expect(panel.locator(".night-predictions")).toContainText("No supported two-candidate margin");
  59 |     }
  60 |     expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  61 |     await panel.getByRole("button", { name: "Close scorecard", exact: true }).click();
  62 |     await expect(panel).toHaveCount(0);
  63 |     await expect(trigger).toBeFocused();
  64 |     expect(page.url()).toBe(originalUrl);
  65 |     await trigger.click();
  66 |     await expect(panel.locator(".night-edition")).toContainText("Standard forecast");
  67 |     await expect(panel.locator(".night-models select")).toHaveValue(defaultId);
  68 |     expect(errors).toEqual([]);
  69 |   });
  70 | }
  71 | 
```