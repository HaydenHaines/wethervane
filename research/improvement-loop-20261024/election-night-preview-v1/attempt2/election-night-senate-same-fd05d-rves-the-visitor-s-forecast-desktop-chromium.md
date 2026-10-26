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
        - link "The races" [ref=e8] [cursor=pointer]:
          - /url: "#races"
        - link "How to read this" [ref=e9] [cursor=pointer]:
          - /url: "#methodology"
        - button "Election night scorecard" [expanded] [ref=e10] [cursor=pointer]
        - generic [ref=e11]: Research preview
    - generic [ref=e13]:
      - tablist "Chamber" [ref=e14]:
        - tab "Senate" [selected] [ref=e15] [cursor=pointer]
        - tab "House" [ref=e16] [cursor=pointer]
      - generic [ref=e17]: Two chambers. Your perspective.
    - region [ref=e18]:
      - generic [ref=e19]:
        - generic [ref=e20]:
          - generic [ref=e21]: An election-night watcher’s guide
          - heading "Election night scorecard" [level=2] [ref=e22]
          - paragraph [ref=e23]: What the model predicts, how wrong it expects to be, and what the results can tell us.
        - button "Close scorecard" [ref=e24] [cursor=pointer]
      - generic [ref=e25]:
        - strong [ref=e26]: Senate · Standard polling
        - generic [ref=e27]: Standard forecast · Dated snapshot
        - paragraph [ref=e28]: Inputs through Tue, 20 Oct 2026 16:21:44 GMT. Forecast issued Tue, 20 Oct 2026 16:27:31 GMT.
        - paragraph [ref=e29]: This is a saved forecast edition. The final election-night evaluation edition and grading rules have not yet been locked.
      - group [ref=e30]:
        - generic "Inspect other model stacks" [active] [ref=e31] [cursor=pointer]
        - generic [ref=e32]:
          - text: Scorecard model
          - combobox "Scorecard model" [ref=e33]:
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
        - paragraph [ref=e34]: Changes here affect only this scorecard. Opening it again starts with the standard forecast.
      - region [ref=e35]:
        - generic [ref=e36]: 1 · Results and predictive power
        - heading "The predictions we can check" [level=3] [ref=e37]
        - generic [ref=e38]:
          - generic [ref=e39]:
            - strong [ref=e40]: "35"
            - generic [ref=e41]: modeled contests
          - generic [ref=e42]:
            - strong [ref=e43]: "35"
            - generic [ref=e44]: with a two-candidate margin
          - generic [ref=e45]:
            - strong [ref=e46]: Not scored
            - generic [ref=e47]: verified election results have not been added
        - heading "Races to watch" [level=4] [ref=e48]
        - paragraph [ref=e49]: These six examples have the least certain winners in the dated standard forecast. The same races stay here when you inspect another stack. Their margins show whether an even win chance also means a close expected race.
        - generic [ref=e50]:
          - article [ref=e51]:
            - generic [ref=e52]:
              - heading "Ohio · Special election" [level=4] [ref=e53]
              - generic [ref=e54]: Awaiting verified result
            - generic [ref=e55]:
              - generic [ref=e56]:
                - term [ref=e57]: Predicted two-candidate margin
                - definition [ref=e58]: Jon Husted +0.6 points
              - generic [ref=e59]:
                - term [ref=e60]: Highest win chance
                - definition [ref=e61]: Jon Husted · 54.4%
              - generic [ref=e62]:
                - term [ref=e63]: Final result / prediction error
                - definition [ref=e64]: Not scored
            - generic [ref=e65]:
              - img "Average margin −0.6 points; middle 90% from −10.0 to +8.9 points. Positive means Sherrod Brown ahead." [ref=e66]:
                - generic [ref=e68]: −20
                - generic [ref=e69]: "0"
                - generic [ref=e70]: +20 points
              - paragraph [ref=e71]:
                - text: "Middle 90%:"
                - strong [ref=e72]: −10.0 to +8.9 points
                - text: .
              - paragraph [ref=e73]: "Positive means Sherrod Brown ahead; negative means Jon Husted ahead. Dot: average margin."
            - group [ref=e74]:
              - generic "Prediction details and what is covered" [ref=e75] [cursor=pointer]
          - article [ref=e76]:
            - generic [ref=e77]:
              - heading "Texas · Regular election" [level=4] [ref=e78]
              - generic [ref=e79]: Awaiting verified result
            - generic [ref=e80]:
              - generic [ref=e81]:
                - term [ref=e82]: Predicted two-candidate margin
                - definition [ref=e83]: KEN PAXTON +1.1 points
              - generic [ref=e84]:
                - term [ref=e85]: Highest win chance
                - definition [ref=e86]: KEN PAXTON · 58.2%
              - generic [ref=e87]:
                - term [ref=e88]: Final result / prediction error
                - definition [ref=e89]: Not scored
            - generic [ref=e90]:
              - img "Average margin −1.1 points; middle 90% from −10.2 to +8.1 points. Positive means JAMES TALARICO ahead." [ref=e91]:
                - generic [ref=e93]: −20
                - generic [ref=e94]: "0"
                - generic [ref=e95]: +20 points
              - paragraph [ref=e96]:
                - text: "Middle 90%:"
                - strong [ref=e97]: −10.2 to +8.1 points
                - text: .
              - paragraph [ref=e98]: "Positive means JAMES TALARICO ahead; negative means KEN PAXTON ahead. Dot: average margin."
            - group [ref=e99]:
              - generic "Prediction details and what is covered" [ref=e100] [cursor=pointer]
          - article [ref=e101]:
            - generic [ref=e102]:
              - heading "Alaska · Regular election" [level=4] [ref=e103]
              - generic [ref=e104]: Awaiting verified result
            - generic [ref=e105]:
              - generic [ref=e106]:
                - term [ref=e107]: Predicted two-candidate margin
                - definition [ref=e108]: Sullivan, Dan S. +1.7 points
              - generic [ref=e109]:
                - term [ref=e110]: Highest win chance
                - definition [ref=e111]: Sullivan, Dan S. · 61.0%
              - generic [ref=e112]:
                - term [ref=e113]: Final result / prediction error
                - definition [ref=e114]: Not scored
            - generic [ref=e115]:
              - img "Average margin −1.7 points; middle 90% from −12.2 to +9.1 points. Positive means Peltola, Mary ahead." [ref=e116]:
                - generic [ref=e118]: −20
                - generic [ref=e119]: "0"
                - generic [ref=e120]: +20 points
              - paragraph [ref=e121]:
                - text: "Middle 90%:"
                - strong [ref=e122]: −12.2 to +9.1 points
                - text: .
              - paragraph [ref=e123]: "Positive means Peltola, Mary ahead; negative means Sullivan, Dan S. ahead. Dot: average margin."
            - group [ref=e124]:
              - generic "Prediction details and what is covered" [ref=e125] [cursor=pointer]
          - article [ref=e126]:
            - generic [ref=e127]:
              - heading "Iowa · Regular election" [level=4] [ref=e128]
              - generic [ref=e129]: Awaiting verified result
            - generic [ref=e130]:
              - generic [ref=e131]:
                - term [ref=e132]: Predicted two-candidate margin
                - definition [ref=e133]: Ashley Hinson +2.3 points
              - generic [ref=e134]:
                - term [ref=e135]: Highest win chance
                - definition [ref=e136]: Ashley Hinson · 66.3%
              - generic [ref=e137]:
                - term [ref=e138]: Final result / prediction error
                - definition [ref=e139]: Not scored
            - generic [ref=e140]:
              - img "Average margin −2.3 points; middle 90% from −11.2 to +6.9 points. Positive means Josh Turek ahead." [ref=e141]:
                - generic [ref=e143]: −20
                - generic [ref=e144]: "0"
                - generic [ref=e145]: +20 points
              - paragraph [ref=e146]:
                - text: "Middle 90%:"
                - strong [ref=e147]: −11.2 to +6.9 points
                - text: .
              - paragraph [ref=e148]: "Positive means Josh Turek ahead; negative means Ashley Hinson ahead. Dot: average margin."
            - group [ref=e149]:
              - generic "Prediction details and what is covered" [ref=e150] [cursor=pointer]
          - article [ref=e151]:
            - generic [ref=e152]:
              - heading "Idaho · Regular election" [level=4] [ref=e153]
              - generic [ref=e154]: Awaiting verified result
            - generic [ref=e155]:
              - generic [ref=e156]:
                - term [ref=e157]: Predicted two-candidate margin
                - definition [ref=e158]: Jim Risch +4.2 points
              - generic [ref=e159]:
                - term [ref=e160]: Highest win chance
                - definition [ref=e161]: Jim Risch · 68.0%
              - generic [ref=e162]:
                - term [ref=e163]: Final result / prediction error
                - definition [ref=e164]: Not scored
            - generic [ref=e165]:
              - img "Average margin −4.2 points; middle 90% from −19.1 to +10.9 points. Positive means Todd Achilles ahead." [ref=e166]:
                - generic [ref=e168]: −20
                - generic [ref=e169]: "0"
                - generic [ref=e170]: +20 points
              - paragraph [ref=e171]:
                - text: "Middle 90%:"
                - strong [ref=e172]: −19.1 to +10.9 points
                - text: .
              - paragraph [ref=e173]: "Positive means Todd Achilles ahead; negative means Jim Risch ahead. Dot: average margin."
            - group [ref=e174]:
              - generic "Prediction details and what is covered" [ref=e175] [cursor=pointer]
          - article [ref=e176]:
            - generic [ref=e177]:
              - heading "Kansas · Regular election" [level=4] [ref=e178]
              - generic [ref=e179]: Awaiting verified result
            - generic [ref=e180]:
              - generic [ref=e181]:
                - term [ref=e182]: Predicted two-candidate margin
                - definition [ref=e183]: Roger Marshall +3.2 points
              - generic [ref=e184]:
                - term [ref=e185]: Highest win chance
                - definition [ref=e186]: Roger Marshall · 70.2%
              - generic [ref=e187]:
                - term [ref=e188]: Final result / prediction error
                - definition [ref=e189]: Not scored
            - generic [ref=e190]:
              - img "Average margin −3.2 points; middle 90% from −13.0 to +7.0 points. Positive means Adam Hamilton ahead." [ref=e191]:
                - generic [ref=e193]: −20
                - generic [ref=e194]: "0"
                - generic [ref=e195]: +20 points
              - paragraph [ref=e196]:
                - text: "Middle 90%:"
                - strong [ref=e197]: −13.0 to +7.0 points
                - text: .
              - paragraph [ref=e198]: "Positive means Adam Hamilton ahead; negative means Roger Marshall ahead. Dot: average margin."
            - group [ref=e199]:
              - generic "Prediction details and what is covered" [ref=e200] [cursor=pointer]
        - group [ref=e201]:
          - generic "All 35 predictions" [ref=e202] [cursor=pointer]
        - paragraph [ref=e203]: When verified results are added, a called winner and a final vote margin will be tracked separately. An incomplete count will not be scored as a final margin.
      - region [ref=e204]:
        - generic [ref=e205]: 2 · Uncertainty
        - heading "What the forecast expects to get wrong" [level=3] [ref=e206]
        - generic [ref=e207]:
          - article [ref=e208]:
            - heading "Wrong top choices" [level=4] [ref=e209]
            - generic [ref=e210]:
              - text: "4.2"
              - generic [ref=e211]: of 35
            - paragraph [ref=e212]: The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.
            - paragraph [ref=e213]: These choices cover the two named principal contenders under each race’s stated assumptions.
          - article [ref=e214]:
            - heading "How far the margins could miss" [level=4] [ref=e215]
            - paragraph [ref=e216]: Awaiting a reviewed uncertainty summary
            - paragraph [ref=e217]: The margin ranges above are available. The model’s expected average margin error and the range for the number of misses have not yet been published.
        - article [ref=e218]:
          - heading "Shared uncertainty in the chamber" [level=4] [ref=e219]
          - paragraph [ref=e220]: Independents unaligned. Existing King and Sanders remain in Democratic caucus; all four modeled independent opposition contenders unassigned, matching stated campaign pledges without treating pledges as certain future conduct.
          - generic [ref=e221]:
            - generic [ref=e222]:
              - term [ref=e223]: Democratic seats
              - definition [ref=e224]: 50.2 average · 46–55 in the middle 90% of simulated elections
            - generic [ref=e225]:
              - term [ref=e226]: Republican seats
              - definition [ref=e227]: 49.0 average · 44–54 in the middle 90% of simulated elections
            - generic [ref=e228]:
              - term [ref=e229]: Other / unassigned seats
              - definition [ref=e230]: 0.7 average · 0–2 in the middle 90% of simulated elections
          - paragraph [ref=e231]: These chamber ranges come from the saved election simulations. Races can miss in the same direction together; they are not independent tests of a national prediction.
        - paragraph [ref=e232]:
          - strong [ref=e233]: 50/50 does not tell us the margin.
          - text: It can mean a narrow contest, or uncertainty about which candidate wins comfortably. Use the separate margin range and close-race chances where available.
      - region [ref=e234]:
        - generic [ref=e235]: The evaluation rubric
        - heading "How the forecast will be judged" [level=3] [ref=e236]
        - generic "A through F scale; no grade assigned" [ref=e237]:
          - generic [ref=e238]: A
          - generic [ref=e239]: B
          - generic [ref=e240]: C
          - generic [ref=e241]: D
          - generic [ref=e242]: F
        - paragraph [ref=e243]:
          - strong [ref=e244]: No letter grade assigned.
          - text: The thresholds, comparison forecasts and rules for combining scores are still under review. They must be published before outcomes are scored.
        - generic [ref=e245]:
          - generic [ref=e246]:
            - term [ref=e247]: Specific predictions
            - definition [ref=e248]: Compare the named candidate margins with verified final results, using the same vote definition. Show the size and direction of each error.
          - generic [ref=e249]:
            - term [ref=e250]: Useful probabilities
            - definition [ref=e251]: Score confident mistakes more heavily and compare performance with simple forecasts on the same races. Predicting everything at 50/50 must not pass as demonstrated skill.
          - generic [ref=e252]:
            - term [ref=e253]: Honest ranges
            - definition [ref=e254]: Check both how often results fall inside the ranges and how wide those ranges were. Very broad ranges do not earn a high grade merely by containing the results.
          - generic [ref=e255]:
            - term [ref=e256]: Shared misses
            - definition [ref=e257]: Check coordinated shifts against the model’s joint predictions. Many misses in one election are not many independent elections.
        - paragraph [ref=e258]: A result having been possible does not excuse poor performance. One surprising result also does not settle whether the model is reliable across elections. Both the individual misses and the comparison scores need to stay visible.
      - group [ref=e259]:
        - generic "Edition, sources and evaluation definitions" [ref=e260] [cursor=pointer]
      - button "Return to the forecast" [ref=e261] [cursor=pointer]
    - tabpanel "Senate" [ref=e262]:
      - main [ref=e263]:
        - generic [ref=e264]:
          - generic [ref=e265]:
            - generic [ref=e266]: 2026 United States Senate · 35 elections
            - heading "The Senate, in perspective." [level=1] [ref=e267]: The Senate,in perspective.
            - paragraph [ref=e268]: Choose the model’s ingredients. Explore the races and chamber together.A closer look at the races that could shape the chamber, with visible uncertainty.
          - generic [ref=e269]:
            - text: THE RESEARCH EDITION
            - strong [ref=e270]: October 20, 2026
            - text: Inputs dated through October 20, 2026
        - group [ref=e271]:
          - generic "Build your perspective Electoral history Expand to customize" [ref=e272] [cursor=pointer]:
            - generic [ref=e273]:
              - generic [ref=e274]: Build your perspective
              - generic [ref=e275]: Electoral history
            - generic [ref=e276]:
              - generic [ref=e277]: Expand to customize
              - generic [aria-hidden] [ref=e278]: ⌄
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
        - generic [ref=e279]:
          - generic [ref=e280]:
            - generic [ref=e281]: Electoral history · 100,000 saved simulations
            - generic [ref=e282]:
              - generic [ref=e283]: Who joins each party?
              - combobox "Who joins each party?" [ref=e284]:
                - option "Independents unaligned" [selected]
                - option "Independents work with Democrats"
                - option "Independents work with Republicans"
          - paragraph [ref=e285]: Each simulation is one possible set of election results. A caucus is a party’s group in the Senate, including independents who work with it. Choose those alliances above; candidate party labels stay the same.
          - region [ref=e286]:
            - generic [ref=e287]:
              - generic [ref=e288]:
                - heading "The path to a majority" [level=2] [ref=e289]
                - paragraph [ref=e290]: 35 elections, ordered by the projected vote-share lead between the two modeled contenders.
              - generic [ref=e291]:
                - generic [ref=e292]: D
                - generic [ref=e293]: R
                - text: Projected control thresholds
            - generic [aria-hidden] [ref=e294]:
              - generic [ref=e295]: Democratic / independent advantage
              - generic [ref=e296]: Republican advantage
            - region "Senate race ribbon, ordered by projected margin. Scroll horizontally for all 35 elections." [ref=e297]:
              - generic [ref=e298]:
                - generic [ref=e299]:
                  - strong [ref=e300]: "34"
                  - generic [ref=e301]: D caucusnot up this year
                - button "Massachusetts, regular election, Safe, projected margin D +32.6 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e302] [cursor=pointer]:
                  - generic [ref=e303]: MA
                  - generic [ref=e304]: ’26
                - button "Delaware, regular election, Safe, projected margin D +22.3 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations." [ref=e305] [cursor=pointer]:
                  - generic [ref=e306]: DE
                  - generic [ref=e307]: ’26
                - button "Oregon, regular election, Safe, projected margin D +22.2 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations." [ref=e308] [cursor=pointer]:
                  - generic [ref=e309]: OR
                  - generic [ref=e310]: ’26
                - button "Rhode Island, regular election, Safe, projected margin D +21.4 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.3% of simulations." [ref=e311] [cursor=pointer]:
                  - generic [ref=e312]: RI
                  - generic [ref=e313]: ’26
                - button "Colorado, regular election, Safe, projected margin D +18.7 pp. At the D control threshold in 0.4% of simulations. At the R control threshold in 0.4% of simulations." [ref=e314] [cursor=pointer]:
                  - generic [ref=e315]: CO
                  - generic [ref=e316]: ’26
                - button "New Mexico, regular election, Likely, projected margin D +13.7 pp. At the D control threshold in 0.8% of simulations. At the R control threshold in 0.8% of simulations." [ref=e317] [cursor=pointer]:
                  - generic [ref=e318]: NM
                  - generic [ref=e319]: ’26
                - button "New Jersey, regular election, Likely, projected margin D +13.7 pp. At the D control threshold in 0.8% of simulations. At the R control threshold in 0.9% of simulations." [ref=e320] [cursor=pointer]:
                  - generic [ref=e321]: NJ
                  - generic [ref=e322]: ’26
                - button "Virginia, regular election, Likely, projected margin D +13.5 pp. At the D control threshold in 0.9% of simulations. At the R control threshold in 0.9% of simulations." [ref=e323] [cursor=pointer]:
                  - generic [ref=e324]: VA
                  - generic [ref=e325]: ’26
                - button "Illinois, regular election, Likely, projected margin D +13.1 pp. At the D control threshold in 0.9% of simulations. At the R control threshold in 0.9% of simulations." [ref=e326] [cursor=pointer]:
                  - generic [ref=e327]: IL
                  - generic [ref=e328]: ’26
                - button "Minnesota, regular election, Likely, projected margin D +6.5 pp. At the D control threshold in 2.5% of simulations. At the R control threshold in 2.6% of simulations." [ref=e329] [cursor=pointer]:
                  - generic [ref=e330]: MN
                  - generic [ref=e331]: ’26
                - button "Georgia, regular election, Likely, projected margin D +5.6 pp. At the D control threshold in 2.9% of simulations. At the R control threshold in 2.9% of simulations." [ref=e332] [cursor=pointer]:
                  - generic [ref=e333]: GA
                  - generic [ref=e334]: ’26
                - button "New Hampshire, regular election, Likely, projected margin D +5.0 pp. At the D control threshold in 3.1% of simulations. At the R control threshold in 3.1% of simulations." [ref=e335] [cursor=pointer]:
                  - generic [ref=e336]: NH
                  - generic [ref=e337]: ’26
                - button "Maine, regular election, Likely, projected margin D +3.6 pp. At the D control threshold in 3.7% of simulations. At the R control threshold in 3.8% of simulations." [ref=e338] [cursor=pointer]:
                  - generic [ref=e339]: ME
                  - generic [ref=e340]: ’26
                - button "Michigan, regular election, Tossup, projected margin D +0.8 pp. At the D control threshold in 5.4% of simulations. At the R control threshold in 5.4% of simulations." [pressed] [ref=e341] [cursor=pointer]:
                  - generic [ref=e342]: MI
                  - generic [ref=e343]: ’26
                - button "North Carolina, regular election, Tossup, projected margin R +0.9 pp. At the D control threshold in 6.6% of simulations. At the R control threshold in 6.6% of simulations." [ref=e344] [cursor=pointer]:
                  - generic [ref=e345]: NC
                  - generic [ref=e346]: ’26
                - button "Iowa, regular election, Likely, projected margin R +11.0 pp. At the D control threshold in 11.2% of simulations. At the R control threshold in 10.1% of simulations." [ref=e347] [cursor=pointer]:
                  - generic [ref=e348]: IA
                  - generic [ref=e349]: ’26
                - button "Texas, regular election, Likely, projected margin R +11.5 pp, projected D control threshold, projected R control threshold. At the D control threshold in 11.1% of simulations. At the R control threshold in 10.0% of simulations." [ref=e350] [cursor=pointer]:
                  - generic [ref=e352]: D51 / R50 · TX
                  - generic [ref=e353]: TX
                  - generic [ref=e354]: ’26
                - button "Ohio, special election, Likely, projected margin R +14.4 pp. At the D control threshold in 9.6% of simulations. At the R control threshold in 8.4% of simulations." [ref=e355] [cursor=pointer]:
                  - generic [ref=e356]: OH
                  - generic [ref=e357]: SP
                - button "Florida, special election, Safe, projected margin R +16.3 pp. At the D control threshold in 8.1% of simulations. At the R control threshold in 7.0% of simulations." [ref=e358] [cursor=pointer]:
                  - generic [ref=e359]: FL
                  - generic [ref=e360]: SP
                - button "Alaska, regular election, Safe, projected margin R +16.8 pp. At the D control threshold in 8.0% of simulations. At the R control threshold in 6.8% of simulations." [ref=e361] [cursor=pointer]:
                  - generic [ref=e362]: AK
                  - generic [ref=e363]: ’26
                - button "Montana, regular election, Safe, projected margin R +18.0 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 5.8% of simulations." [ref=e364] [cursor=pointer]:
                  - generic [ref=e365]: MT
                  - generic [ref=e366]: ’26
                - button "Kansas, regular election, Safe, projected margin R +19.4 pp. At the D control threshold in 5.7% of simulations. At the R control threshold in 4.9% of simulations." [ref=e367] [cursor=pointer]:
                  - generic [ref=e368]: KS
                  - generic [ref=e369]: ’26
                - button "Louisiana, regular election, Safe, projected margin R +19.9 pp. At the D control threshold in 5.4% of simulations. At the R control threshold in 4.6% of simulations." [ref=e370] [cursor=pointer]:
                  - generic [ref=e371]: LA
                  - generic [ref=e372]: ’26
                - button "South Carolina, regular election, Safe, projected margin R +21.1 pp. At the D control threshold in 4.5% of simulations. At the R control threshold in 3.8% of simulations." [ref=e373] [cursor=pointer]:
                  - generic [ref=e374]: SC
                  - generic [ref=e375]: ’26
                - button "Nebraska, regular election, Safe, projected margin R +23.7 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.6% of simulations." [ref=e376] [cursor=pointer]:
                  - generic [ref=e377]: NE
                  - generic [ref=e378]: ’26
                - button "Mississippi, regular election, Safe, projected margin R +25.9 pp. At the D control threshold in 2.2% of simulations. At the R control threshold in 1.8% of simulations." [ref=e379] [cursor=pointer]:
                  - generic [ref=e380]: MS
                  - generic [ref=e381]: ’26
                - button "Alabama, regular election, Safe, projected margin R +28.4 pp. At the D control threshold in 1.5% of simulations. At the R control threshold in 1.2% of simulations." [ref=e382] [cursor=pointer]:
                  - generic [ref=e383]: AL
                  - generic [ref=e384]: ’26
                - button "Kentucky, regular election, Safe, projected margin R +28.6 pp. At the D control threshold in 1.5% of simulations. At the R control threshold in 1.2% of simulations." [ref=e385] [cursor=pointer]:
                  - generic [ref=e386]: KY
                  - generic [ref=e387]: ’26
                - button "Oklahoma, regular election, Safe, projected margin R +32.4 pp. At the D control threshold in 0.8% of simulations. At the R control threshold in 0.6% of simulations." [ref=e388] [cursor=pointer]:
                  - generic [ref=e389]: OK
                  - generic [ref=e390]: ’26
                - button "South Dakota, regular election, Safe, projected margin R +32.5 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.6% of simulations." [ref=e391] [cursor=pointer]:
                  - generic [ref=e392]: SD
                  - generic [ref=e393]: ’26
                - button "Tennessee, regular election, Safe, projected margin R +32.7 pp. At the D control threshold in 0.7% of simulations. At the R control threshold in 0.6% of simulations." [ref=e394] [cursor=pointer]:
                  - generic [ref=e395]: TN
                  - generic [ref=e396]: ’26
                - button "Arkansas, regular election, Safe, projected margin R +33.9 pp. At the D control threshold in 0.6% of simulations. At the R control threshold in 0.5% of simulations." [ref=e397] [cursor=pointer]:
                  - generic [ref=e398]: AR
                  - generic [ref=e399]: ’26
                - button "Idaho, regular election, Safe, projected margin R +39.9 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.2% of simulations." [ref=e400] [cursor=pointer]:
                  - generic [ref=e401]: ID
                  - generic [ref=e402]: ’26
                - button "Wyoming, regular election, Safe, projected margin R +44.5 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e403] [cursor=pointer]:
                  - generic [ref=e404]: WY
                  - generic [ref=e405]: ’26
                - button "West Virginia, regular election, Safe, projected margin R +44.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations." [ref=e406] [cursor=pointer]:
                  - generic [ref=e407]: WV
                  - generic [ref=e408]: ’26
                - generic [ref=e409]:
                  - strong [ref=e410]: "31"
                  - generic [ref=e411]: R caucusnot up this year
            - paragraph [ref=e412]: Read inward from either party’s end. Its marker identifies the last race needed for control in this projected order, after seats not up for election and races guaranteed to join that party’s group. D = Democratic; R = Republican. “SP” marks a special election to fill an unfinished term.
            - generic [ref=e413]:
              - generic [ref=e414]:
                - generic [ref=e415]: D
                - generic [ref=e416]:
                  - strong [ref=e417]:
                    - button "Texas" [ref=e418] [cursor=pointer]
                  - generic [ref=e419]: 51 caucus seats for control · projected margin −11.5 pp
                  - generic [ref=e420]:
                    - text: "Most frequent in simulations:"
                    - button "Iowa" [ref=e421] [cursor=pointer]
                    - text: · 11.2%
              - generic [ref=e422]:
                - generic [ref=e423]: R
                - generic [ref=e424]:
                  - strong [ref=e425]:
                    - button "Texas" [ref=e426] [cursor=pointer]
                  - generic [ref=e427]: 50 caucus seats for control · projected margin +11.5 pp
                  - generic [ref=e428]:
                    - text: "Most frequent in simulations:"
                    - button "Iowa" [ref=e429] [cursor=pointer]
                    - text: · 10.1%
            - generic [ref=e430]:
              - strong [ref=e431]: Michigan · D +0.8 pp
              - generic [ref=e432]: At the D control threshold in 5.4% of simulations.
              - generic [ref=e433]: At the R control threshold in 5.4% of simulations.
            - paragraph [ref=e434]: "A margin is the difference in vote shares: 52% versus 48% is a 4 percentage-point (pp) lead. The frequencies above count how often a race occupies the control-threshold rank across simulated elections, including simulations where that party falls short. They are not the chance that flipping that race alone would change control."
            - group [ref=e435]:
              - generic "How to read the thresholds" [ref=e436] [cursor=pointer]
        - region "Senate race explorer" [ref=e437]:
          - generic [ref=e438]:
            - generic [ref=e439]:
              - generic [ref=e440]:
                - generic [ref=e441]:
                  - heading "A nation of distinct races" [level=2] [ref=e442]
                  - generic [ref=e443]: Select a state to explore its election.
                - generic [ref=e444]: 2026 Senate
              - group "United States Senate races. Use Tab to select a state with an election." [ref=e445]:
                - button "New Mexico, 1 election, Likely" [ref=e446] [cursor=pointer]
                - button "South Dakota, 1 election, Safe" [ref=e447] [cursor=pointer]
                - generic "California, no Senate election" [ref=e448]
                - button "Kentucky, 1 election, Safe" [ref=e449] [cursor=pointer]
                - button "Alabama, 1 election, Safe" [ref=e450] [cursor=pointer]
                - button "Georgia, 1 election, Likely" [ref=e451] [cursor=pointer]
                - button "Arkansas, 1 election, Safe" [ref=e452] [cursor=pointer]
                - generic "Pennsylvania, no Senate election" [ref=e453]
                - generic "Missouri, no Senate election" [ref=e454]
                - button "Colorado, 1 election, Safe" [ref=e455] [cursor=pointer]
                - generic "Utah, no Senate election" [ref=e456]
                - button "Oklahoma, 1 election, Safe" [ref=e457] [cursor=pointer]
                - button "Tennessee, 1 election, Safe" [ref=e458] [cursor=pointer]
                - button "Wyoming, 1 election, Safe" [ref=e459] [cursor=pointer]
                - generic "New York, no Senate election" [ref=e460]
                - generic "Indiana, no Senate election" [ref=e461]
                - button "Kansas, 1 election, Safe" [ref=e462] [cursor=pointer]
                - button "Idaho, 1 election, Safe" [ref=e463] [cursor=pointer]
                - button "Alaska, 1 election, Safe" [ref=e464] [cursor=pointer]
                - generic "Nevada, no Senate election" [ref=e465]
                - button "Illinois, 1 election, Likely" [ref=e466] [cursor=pointer]
                - generic "Vermont, no Senate election" [ref=e467]
                - button "New Jersey, 1 election, Likely" [ref=e468] [cursor=pointer]
                - generic "North Dakota, no Senate election" [ref=e469]
                - button "Iowa, 1 election, Likely" [ref=e470] [cursor=pointer]
                - button "South Carolina, 1 election, Safe" [ref=e471] [cursor=pointer]
                - generic "Arizona, no Senate election" [ref=e472]
                - button "Delaware, 1 election, Safe" [ref=e473] [cursor=pointer]
                - generic "District of Columbia, no Senate election" [ref=e474]
                - generic "Connecticut, no Senate election" [ref=e475]
                - button "New Hampshire, 1 election, Likely" [ref=e476] [cursor=pointer]
                - button "Nebraska, 1 election, Safe" [ref=e477] [cursor=pointer]
                - generic "Washington, no Senate election" [ref=e478]
                - button "Texas, 1 election, Likely" [ref=e479] [cursor=pointer]
                - button "Ohio, 1 election, Likely" [ref=e480] [cursor=pointer]
                - generic "Wisconsin, no Senate election" [ref=e481]
                - button "Oregon, 1 election, Safe" [ref=e482] [cursor=pointer]
                - button "Mississippi, 1 election, Safe" [ref=e483] [cursor=pointer]
                - button "North Carolina, 1 election, Tossup" [ref=e484] [cursor=pointer]
                - button "Virginia, 1 election, Likely" [ref=e485] [cursor=pointer]
                - button "West Virginia, 1 election, Safe" [ref=e486] [cursor=pointer]
                - button "Louisiana, 1 election, Safe" [ref=e487] [cursor=pointer]
                - button "Michigan, 1 election, Tossup" [pressed] [ref=e488] [cursor=pointer]
                - button "Massachusetts, 1 election, Safe" [ref=e489] [cursor=pointer]
                - button "Florida, 1 election, Safe" [ref=e490] [cursor=pointer]
                - button "Montana, 1 election, Safe" [ref=e491] [cursor=pointer]
                - button "Minnesota, 1 election, Likely" [ref=e492] [cursor=pointer]
                - generic "Maryland, no Senate election" [ref=e493]
                - button "Maine, 1 election, Likely" [ref=e494] [cursor=pointer]
                - generic "Hawaii, no Senate election" [ref=e495]
                - button "Rhode Island, 1 election, Safe" [ref=e496] [cursor=pointer]
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
                - generic [ref=e497]: Alaska and Hawaii shown in insets; scale varies.
              - generic [ref=e498]:
                - generic [ref=e499]: Safe D
                - generic [ref=e501]: Likely D
                - generic [ref=e503]: Safe R
                - generic [ref=e505]: Likely R
                - generic [ref=e507]: Safe I
                - generic [ref=e509]: Likely I
                - generic [ref=e511]: Tossup
              - generic [ref=e513]:
                - generic [ref=e514]: "Ratings use the favorite’s chance of winning: Safe above 90%; Likely above 60% through 90%; Tossup 60% or lower (40–60% each in a two-outcome race). Safe does not mean certain."
                - generic [ref=e515]: "D = Democratic · R = Republican · I = Independent. Gray: no election. State area does not represent votes."
            - region [ref=e516]:
              - generic [ref=e518]:
                - heading "Who could control the chamber?" [level=2] [ref=e519]
                - generic [ref=e520]: Independents unaligned · D needs 51; R needs 50
              - generic [ref=e521]:
                - generic [ref=e522]:
                  - generic [ref=e523]: Democratic control
                  - generic [ref=e525]: 9.4%
                - generic [ref=e526]:
                  - generic [ref=e527]: Republican control
                  - generic [ref=e529]: 89.4%
                - generic [ref=e530]:
                  - generic [ref=e531]: Control depends on alliances
                  - generic [ref=e533]: 1.3%
              - paragraph [ref=e538]: King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
              - paragraph [ref=e539]: A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.
              - generic [ref=e540]:
                - generic [ref=e541]:
                  - text: "D caucus: 46.8 expected seats"
                  - generic [ref=e542]: (42–52, modeled 90% range)
                - generic [ref=e543]:
                  - text: "R caucus: 53.0 expected seats"
                  - generic [ref=e544]: (48–58, modeled 90% range)
                - generic [ref=e545]:
                  - text: "Unaligned: 0.1 expected seats"
                  - generic [ref=e546]: (0–1, modeled 90% range)
              - paragraph [ref=e547]: Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.
            - region [ref=e548]:
              - heading "What has been tested?" [level=2] [ref=e549]
              - paragraph [ref=e550]: The 2026 election has not happened yet. Select 2018 or 2022 to see how historical reconstructions compare with actual results. Those views use a different model of possible winners, so their scores do not measure the accuracy of this 2026 two-contender forecast.
              - group [ref=e551]:
                - generic "Scope and limitations" [ref=e552] [cursor=pointer]
            - region [ref=e553]:
              - generic [ref=e554]:
                - heading "The races" [level=2] [ref=e555]
                - generic [ref=e556]: 10 of 35 elections
              - paragraph [ref=e557]: The favorite is the candidate or group with the highest modeled chance of winning. “Most competitive” puts the least certain favorites first.
              - generic [ref=e558]:
                - generic [ref=e559]:
                  - generic [aria-hidden] [ref=e560]: ⌕
                  - textbox "Search states or candidates" [ref=e561]:
                    - /placeholder: Find a state or candidate
                - combobox "Sort races" [ref=e562]:
                  - option "Most competitive" [selected]
                  - option "State A–Z"
              - table [ref=e563]:
                - rowgroup [ref=e564]:
                  - row [ref=e565]:
                    - columnheader "State / election" [ref=e566]
                    - columnheader "Favorite" [ref=e567]
                    - columnheader "Chance to win" [ref=e568]
                    - columnheader "Outlook" [ref=e569]
                - rowgroup [ref=e570]:
                  - row [ref=e571]:
                    - cell [ref=e572]:
                      - button "Michigan Regular election" [pressed] [ref=e573] [cursor=pointer]:
                        - text: Michigan
                        - generic [ref=e574]: Regular election
                    - cell "Abdul El-Sayed Democratic" [ref=e575]:
                      - text: Abdul El-Sayed
                      - generic [ref=e576]: Democratic
                    - cell "52.8%" [ref=e577]
                    - cell "Tossup" [ref=e581]
                  - row [ref=e582]:
                    - cell [ref=e583]:
                      - button "North Carolina Regular election" [ref=e584] [cursor=pointer]:
                        - text: North Carolina
                        - generic [ref=e585]: Regular election
                    - cell "Michael Whatley Republican" [ref=e586]:
                      - text: Michael Whatley
                      - generic [ref=e587]: Republican
                    - cell "53.3%" [ref=e588]
                    - cell "Tossup" [ref=e592]
                  - row [ref=e593]:
                    - cell [ref=e594]:
                      - button "Maine Regular election" [ref=e595] [cursor=pointer]:
                        - text: Maine
                        - generic [ref=e596]: Regular election
                    - cell "Jackson, Troy D. Democratic" [ref=e597]:
                      - text: Jackson, Troy D.
                      - generic [ref=e598]: Democratic
                    - cell "62.0%" [ref=e599]
                    - cell "Likely" [ref=e603]
                  - row [ref=e604]:
                    - cell [ref=e605]:
                      - button "New Hampshire Regular election" [ref=e606] [cursor=pointer]:
                        - text: New Hampshire
                        - generic [ref=e607]: Regular election
                    - cell "Chris Pappas Democratic" [ref=e608]:
                      - text: Chris Pappas
                      - generic [ref=e609]: Democratic
                    - cell "66.7%" [ref=e610]
                    - cell "Likely" [ref=e614]
                  - row [ref=e615]:
                    - cell [ref=e616]:
                      - button "Georgia Regular election" [ref=e617] [cursor=pointer]:
                        - text: Georgia
                        - generic [ref=e618]: Regular election
                    - cell "Jon Ossoff Democratic" [ref=e619]:
                      - text: Jon Ossoff
                      - generic [ref=e620]: Democratic
                    - cell "68.5%" [ref=e621]
                    - cell "Likely" [ref=e625]
                  - row [ref=e626]:
                    - cell [ref=e627]:
                      - button "Minnesota Regular election" [ref=e628] [cursor=pointer]:
                        - text: Minnesota
                        - generic [ref=e629]: Regular election
                    - cell "Peggy Flanagan Democratic" [ref=e630]:
                      - text: Peggy Flanagan
                      - generic [ref=e631]: Democratic
                    - cell "71.3%" [ref=e632]
                    - cell "Likely" [ref=e636]
                  - row [ref=e637]:
                    - cell [ref=e638]:
                      - button "Iowa Regular election" [ref=e639] [cursor=pointer]:
                        - text: Iowa
                        - generic [ref=e640]: Regular election
                    - cell "Ashley Hinson Republican" [ref=e641]:
                      - text: Ashley Hinson
                      - generic [ref=e642]: Republican
                    - cell "82.3%" [ref=e643]
                    - cell "Likely" [ref=e647]
                  - row [ref=e648]:
                    - cell [ref=e649]:
                      - button "Texas Regular election" [ref=e650] [cursor=pointer]:
                        - text: Texas
                        - generic [ref=e651]: Regular election
                    - cell "KEN PAXTON Republican" [ref=e652]:
                      - text: KEN PAXTON
                      - generic [ref=e653]: Republican
                    - cell "83.2%" [ref=e654]
                    - cell "Likely" [ref=e658]
                  - row [ref=e659]:
                    - cell [ref=e660]:
                      - button "Illinois Regular election" [ref=e661] [cursor=pointer]:
                        - text: Illinois
                        - generic [ref=e662]: Regular election
                    - cell "Juliana Stratton Democratic" [ref=e663]:
                      - text: Juliana Stratton
                      - generic [ref=e664]: Democratic
                    - cell "86.0%" [ref=e665]
                    - cell "Likely" [ref=e669]
                  - row [ref=e670]:
                    - cell [ref=e671]:
                      - button "Virginia Regular election" [ref=e672] [cursor=pointer]:
                        - text: Virginia
                        - generic [ref=e673]: Regular election
                    - cell "Mark R. Warner Democratic" [ref=e674]:
                      - text: Mark R. Warner
                      - generic [ref=e675]: Democratic
                    - cell "86.5%" [ref=e676]
                    - cell "Likely" [ref=e680]
              - button "See all 35 races" [ref=e681] [cursor=pointer]:
                - text: See all 35 races
                - generic [aria-hidden] [ref=e682]: ⌄
          - complementary "Selected race detail" [ref=e683]:
            - generic [ref=e684]:
              - generic [ref=e685]: Race in focus
              - generic [ref=e686]: Tossup
            - heading "Michigan" [level=2] [ref=e687]
            - generic [ref=e688]: U.S. Senate · Regular election
            - separator [ref=e689]
            - heading "Chance of winning" [level=3] [ref=e690]
            - paragraph [ref=e691]: A 70% chance means winning about 70 out of 100 elections simulated by this model. It does not mean winning 70% of the votes.
            - generic [ref=e692]:
              - generic [ref=e694]:
                - generic [ref=e695]:
                  - generic [ref=e696]: Abdul El-Sayed
                  - generic [ref=e697]: Democratic
                - strong [ref=e698]: 52.8%
              - generic [ref=e702]:
                - generic [ref=e703]:
                  - generic [ref=e704]: Mike Rogers
                  - generic [ref=e705]: Republican
                - strong [ref=e706]: 47.2%
            - separator [ref=e709]
            - heading "Vote share among these two contenders" [level=3] [ref=e710]
            - generic [ref=e711]:
              - generic [ref=e712]:
                - generic [ref=e713]:
                  - generic [ref=e714]: El-Sayed
                  - strong [ref=e715]: 50.4%
                - 'generic "Abdul El-Sayed: mean 50.4%, Modeled 90% range 39.6% to 61.1%" [ref=e716]'
                - paragraph [ref=e719]: "Modeled 90% range: 39.6%–61.1%Share of votes for the two named principal contenders"
              - generic [ref=e720]:
                - generic [ref=e721]:
                  - generic [ref=e722]: Rogers
                  - strong [ref=e723]: 49.6%
                - 'generic "Mike Rogers: mean 49.6%, Modeled 90% range 38.9% to 60.4%" [ref=e724]'
                - paragraph [ref=e727]: "Modeled 90% range: 38.9%–60.4%Share of votes for the two named principal contenders"
            - paragraph [ref=e728]: The shares divide votes between these two contenders only; votes for other candidates are excluded. The range runs from the 5th to the 95th percentile of simulated shares—the middle 90%. It is a model estimate, not a poll’s margin of error or a guarantee.
            - paragraph [ref=e729]: Probabilities of 0% or 100% reflect the saved simulation, not certainty.
            - separator [ref=e730]
            - generic [ref=e732]:
              - text: Selected model setup
              - strong [ref=e733]: Electoral history
            - paragraph [ref=e734]: This setup uses electoral history without polls.
            - group [ref=e735]:
              - generic "Race assumptions" [ref=e736] [cursor=pointer]
            - separator [ref=e737]
            - heading "Candidate and result sources" [level=3] [ref=e738]
            - generic [ref=e739]:
              - link "Abdul El-Sayed ↗" [ref=e741] [cursor=pointer]:
                - /url: https://www.isabellacounty.org/departments/county-clerk/services/general-election-information/
              - link "Mike Rogers ↗" [ref=e743] [cursor=pointer]:
                - /url: https://www.isabellacounty.org/departments/county-clerk/services/general-election-information/
            - paragraph [ref=e744]: "Forecast date: October 20, 2026Inputs dated through: October 20, 2026Saved research forecast; new polls are not added automatically."
        - region [ref=e745]:
          - generic [ref=e746]:
            - generic [ref=e747]: From evidence to possible elections
            - heading "How this forecast works." [level=2] [ref=e748]
            - paragraph [ref=e749]: A model is a set of mathematical assumptions connecting evidence to possible results. Your choices change those assumptions and, when available, load the matching saved forecast.
            - link "Download this forecast’s data ↓" [ref=e750] [cursor=pointer]:
              - /url: /data/explorer/current-fundamentals.json
            - paragraph [ref=e751]: "The download is a JSON file: structured data for analysis, with source identifiers and limitations."
          - generic [ref=e752]:
            - generic [ref=e753]:
              - heading "1. Start with a baseline" [level=3] [ref=e754]
              - paragraph [ref=e755]: Electoral history supplies a starting estimate, often called “fundamentals.” The builder also offers a historical winner reference model. A component is one part of the model; the chosen combination is your model setup or “stack.”
            - generic [ref=e756]:
              - heading "2. Add the selected evidence" [level=3] [ref=e757]
              - paragraph [ref=e758]:
                - text: This setup uses no polls. Polls measure responses at a particular time. The model allows for measurement errors and changes before election day.
                - link "Read the poll selection rules below." [ref=e759] [cursor=pointer]:
                  - /url: "#poll-selection"
            - generic [ref=e760]:
              - heading "3. Simulate elections together" [level=3] [ref=e761]
              - paragraph [ref=e762]: A simulation is one possible set of election results generated by the model. Shared uncertainty allows many races to move together—for example, if polls miss in the same direction. Win chances and chamber control summarize these simulated elections, rather than assuming each race is unrelated.
            - generic [ref=e763]:
              - heading "4. Explore what changes" [level=3] [ref=e764]
              - paragraph [ref=e765]: Changing a choice loads a saved result if one exists. Some combinations require calculating several model parts together again (a “joint refit”); the website explains when that work is needed. A sensitivity test changes assumptions without adding evidence. “Runs successfully,” “experimental,” and “tested against election results” describe different things.
            - generic [ref=e766]:
              - heading "What these numbers cover" [level=3] [ref=e767]
              - paragraph [ref=e768]: The 2026 preview models only the two named contenders in each race. Its vote shares divide votes between those two; other candidates’ shares and chances of winning are outside this model.
            - generic [ref=e769]:
              - heading "Make fair comparisons" [level=3] [ref=e770]
              - paragraph [ref=e771]: Compare the same year, possible winners and assumptions about independent senators. A change from 40% to 45% is 5 percentage points (“pp”), not a 5% relative increase. Historical scores describe those elections and do not establish the accuracy of the current forecast.
        - region [ref=e772]:
          - generic [ref=e773]: The evidence behind this view
          - heading "Why some polls are included—and others are not" [level=2] [ref=e774]
          - paragraph [ref=e775]: This model setup uses electoral history without polling. No polls are included. Choose a polling component in “Build your perspective” to explore a saved forecast that uses polls.
        - generic [ref=e776]:
          - generic [ref=e777]: ballotmancy
          - generic [ref=e778]: Independent research. Visible assumptions.2026 Senate research preview · October 20, 2026
  - alert [ref=e779]
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