# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: explanations.spec.ts >> refreshed poll explanations connect counts, date bounds, sample Ns and provider proxies
- Location: tests/site/explanations.spec.ts:3:5

# Error details

```
Error: expect(locator).toContainText(expected) failed

Locator: locator('#poll-selection').locator('.poll-record-list li').filter({ hasText: 'Data for Progress' }).filter({ has: locator('#poll-selection').locator('.poll-record-status.included') })
Expected substring: "Exact publication date not verified."
Timeout: 5000ms
Error: element(s) not found

Call log:
  - Expect "toContainText" locator('#poll-selection').locator('.poll-record-list li').filter({ hasText: 'Data for Progress' }).filter({ has: locator('#poll-selection').locator('.poll-record-status.included') }) with timeout 5000ms
  - waiting for locator('#poll-selection').locator('.poll-record-list li').filter({ hasText: 'Data for Progress' }).filter({ has: locator('#poll-selection').locator('.poll-record-status.included') })

```

```yaml
- link "Skip to the races":
  - /url: "#races"
- banner:
  - link "Ballotmancy home":
    - /url: /
    - text: ballotmancy
  - navigation "Main navigation":
    - link "The races":
      - /url: "#races"
    - link "How to read this":
      - /url: "#methodology"
    - text: Research preview
- tablist "Chamber":
  - tab "Senate" [selected]
  - tab "House"
- text: Two chambers. Your perspective.
- tabpanel "Senate":
  - main:
    - text: 2026 United States Senate · 35 elections
    - heading "The Senate, in perspective." [level=1]
    - paragraph: Choose the model’s ingredients. Explore the races and chamber together. A closer look at the races that could shape the chamber, with visible uncertainty.
    - group: Input and publication times
    - text: THE RESEARCH EDITION
    - strong: October 26, 2026
    - text: Inputs dated through October 26, 2026
    - group "Build your perspective": Build your perspective Standard polling Expand to customize
    - text: Standard polling · 40,000 saved simulations Who joins each party?
    - combobox "Who joins each party?":
      - option "Independents unaligned" [selected]
      - option "Independents work with Democrats"
      - option "Independents work with Republicans"
    - paragraph: Each simulation is one possible set of election results. A caucus is a party’s group in the Senate, including independents who work with it. Choose those alliances above; candidate party labels stay the same.
    - region "The path to a majority":
      - heading "The path to a majority" [level=2]
      - paragraph: 35 elections, ordered by the projected vote-share lead between the two modeled contenders.
      - text: D R Projected control thresholds
      - region "Senate race ribbon, ordered by projected margin. Scroll horizontally for all 35 elections.":
        - strong: "34"
        - text: D caucus not up this year
        - button "Rhode Island, regular election, Safe, projected margin D +31.4 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations.": RI ’26
        - button "Massachusetts, regular election, Safe, projected margin D +30.8 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations.": MA ’26
        - button "Oregon, regular election, Safe, projected margin D +28.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations.": OR ’26
        - button "Delaware, regular election, Safe, projected margin D +28.8 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations.": DE ’26
        - button "Colorado, regular election, Safe, projected margin D +25.4 pp. At the D control threshold in 0.2% of simulations. At the R control threshold in 0.2% of simulations.": CO ’26
        - button "New Jersey, regular election, Safe, projected margin D +20.5 pp. At the D control threshold in 0.5% of simulations. At the R control threshold in 0.5% of simulations.": NJ ’26
        - button "Virginia, regular election, Safe, projected margin D +20.4 pp. At the D control threshold in 0.5% of simulations. At the R control threshold in 0.5% of simulations.": VA ’26
        - button "New Mexico, regular election, Safe, projected margin D +17.7 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations.": NM ’26
        - button "Illinois, regular election, Safe, projected margin D +16.5 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations.": IL ’26
        - button "Georgia, regular election, Safe, projected margin D +8.2 pp. At the D control threshold in 1.6% of simulations. At the R control threshold in 1.7% of simulations.": GA ’26
        - button "North Carolina, regular election, Safe, projected margin D +8.1 pp. At the D control threshold in 1.3% of simulations. At the R control threshold in 1.4% of simulations.": NC ’26
        - button "New Hampshire, regular election, Safe, projected margin D +7.7 pp. At the D control threshold in 2.1% of simulations. At the R control threshold in 2.1% of simulations.": NH ’26
        - button "Minnesota, regular election, Likely, projected margin D +4.8 pp. At the D control threshold in 5.2% of simulations. At the R control threshold in 5.2% of simulations.": MN ’26
        - button "Michigan, regular election, Likely, projected margin D +2.6 pp. At the D control threshold in 8.9% of simulations. At the R control threshold in 8.6% of simulations.": MI ’26
        - button "Maine, regular election, Likely, projected margin D +1.8 pp. At the D control threshold in 9.9% of simulations. At the R control threshold in 9.3% of simulations.": ME ’26
        - button "Ohio, special election, Tossup, projected margin R +0.3 pp. At the D control threshold in 13.1% of simulations. At the R control threshold in 11.9% of simulations.": OH SP
        - button "Texas, regular election, Tossup, projected margin R +1.1 pp, projected D control threshold, projected R control threshold. At the D control threshold in 13.0% of simulations. At the R control threshold in 11.5% of simulations." [pressed]: TX ’26
        - button "Alaska, regular election, Likely, projected margin R +2.1 pp. At the D control threshold in 10.1% of simulations. At the R control threshold in 8.8% of simulations.": AK ’26
        - button "Iowa, regular election, Likely, projected margin R +2.5 pp. At the D control threshold in 11.4% of simulations. At the R control threshold in 9.8% of simulations.": IA ’26
        - button "Kansas, regular election, Likely, projected margin R +3.7 pp. At the D control threshold in 8.7% of simulations. At the R control threshold in 7.4% of simulations.": KS ’26
        - button "South Carolina, regular election, Likely, projected margin R +5.4 pp. At the D control threshold in 6.0% of simulations. At the R control threshold in 5.0% of simulations.": SC ’26
        - button "Nebraska, regular election, Likely, projected margin R +5.7 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 4.7% of simulations.": NE ’26
        - button "Florida, special election, Safe, projected margin R +8.1 pp. At the D control threshold in 2.3% of simulations. At the R control threshold in 1.8% of simulations.": FL SP
        - button "Montana, regular election, Safe, projected margin R +8.3 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.4% of simulations.": MT ’26
        - button "Idaho, regular election, Likely, projected margin R +8.8 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 2.8% of simulations.": ID ’26
        - button "Louisiana, regular election, Likely, projected margin R +12.0 pp. At the D control threshold in 2.3% of simulations. At the R control threshold in 2.0% of simulations.": LA ’26
        - button "Mississippi, regular election, Safe, projected margin R +13.4 pp. At the D control threshold in 1.1% of simulations. At the R control threshold in 0.9% of simulations.": MS ’26
        - button "Arkansas, regular election, Safe, projected margin R +17.1 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.2% of simulations.": AR ’26
        - button "South Dakota, regular election, Safe, projected margin R +17.9 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations.": SD ’26
        - button "Kentucky, regular election, Safe, projected margin R +21.0 pp. At the D control threshold in 0.6% of simulations. At the R control threshold in 0.5% of simulations.": KY ’26
        - button "Alabama, regular election, Safe, projected margin R +21.5 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations.": AL ’26
        - button "Oklahoma, regular election, Safe, projected margin R +25.1 pp. At the D control threshold in 0.3% of simulations. At the R control threshold in 0.3% of simulations.": OK ’26
        - button "Tennessee, regular election, Safe, projected margin R +26.2 pp. At the D control threshold in 0.0% of simulations. At the R control threshold in 0.0% of simulations.": TN ’26
        - button "West Virginia, regular election, Safe, projected margin R +37.2 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.1% of simulations.": WV ’26
        - button "Wyoming, regular election, Safe, projected margin R +37.9 pp. At the D control threshold in 0.1% of simulations. At the R control threshold in 0.0% of simulations.": WY ’26
        - strong: "31"
        - text: R caucus not up this year
      - paragraph: Read inward from either party’s end. Its marker identifies the last race needed for control in this projected order, after seats not up for election and races guaranteed to join that party’s group. D = Democratic; R = Republican. “SP” marks a special election to fill an unfinished term.
      - text: D
      - strong:
        - button "Texas"
      - text: "51 caucus seats for control · projected margin −1.1 pp Most frequent in simulations:"
      - button "Ohio"
      - text: · 13.1% R
      - strong:
        - button "Texas"
      - text: "50 caucus seats for control · projected margin +1.1 pp Most frequent in simulations:"
      - button "Ohio"
      - text: · 11.9%
      - strong: Texas · R +1.1 pp
      - text: At the D control threshold in 13.0% of simulations. At the R control threshold in 11.5% of simulations.
      - paragraph: "A margin is the difference in vote shares: 52% versus 48% is a 4 percentage-point (pp) lead. The frequencies above count how often a race occupies the control-threshold rank across simulated elections, including simulations where that party falls short. They are not the chance that flipping that race alone would change control."
      - group: How to read the thresholds
    - region "Senate race explorer":
      - heading "A nation of distinct races" [level=2]
      - text: Select a state to explore its election. 2026 Senate
      - group "United States Senate races. Use Tab to select a state with an election.":
        - button "New Mexico, 1 election, Safe"
        - button "South Dakota, 1 election, Safe"
        - button "Kentucky, 1 election, Safe"
        - button "Alabama, 1 election, Safe"
        - button "Georgia, 1 election, Safe"
        - button "Arkansas, 1 election, Safe"
        - button "Colorado, 1 election, Safe"
        - button "Oklahoma, 1 election, Safe"
        - button "Tennessee, 1 election, Safe"
        - button "Wyoming, 1 election, Safe"
        - button "Kansas, 1 election, Likely"
        - button "Idaho, 1 election, Likely"
        - button "Alaska, 1 election, Likely"
        - button "Illinois, 1 election, Safe"
        - button "New Jersey, 1 election, Safe"
        - button "Iowa, 1 election, Likely"
        - button "South Carolina, 1 election, Likely"
        - button "Delaware, 1 election, Safe"
        - button "New Hampshire, 1 election, Safe"
        - button "Nebraska, 1 election, Likely"
        - button "Texas, 1 election, Tossup" [pressed]
        - button "Ohio, 1 election, Tossup"
        - button "Oregon, 1 election, Safe"
        - button "Mississippi, 1 election, Safe"
        - button "North Carolina, 1 election, Safe"
        - button "Virginia, 1 election, Safe"
        - button "West Virginia, 1 election, Safe"
        - button "Louisiana, 1 election, Likely"
        - button "Michigan, 1 election, Likely"
        - button "Massachusetts, 1 election, Safe"
        - button "Florida, 1 election, Safe"
        - button "Montana, 1 election, Safe"
        - button "Minnesota, 1 election, Likely"
        - button "Maine, 1 election, Likely"
        - button "Rhode Island, 1 election, Safe"
        - text: NM SD CA KY AL GA AR PA MO CO UT OK TN WY NY IN KS ID AK NV IL VT ND IA SC AZ NH NE WA TX OH WI OR MS NC VA WV LA MI FL MT MN ME HI Alaska and Hawaii shown in insets; scale varies.
      - text: "Safe D Likely D Safe R Likely R Safe I Likely I Tossup Ratings use the favorite’s chance of winning: Safe above 90%; Likely above 60% through 90%; Tossup 60% or lower (40–60% each in a two-outcome race). Safe does not mean certain. D = Democratic · R = Republican · I = Independent. Gray: no election. State area does not represent votes."
      - region "Who could control the chamber?":
        - heading "Who could control the chamber?" [level=2]
        - text: Independents unaligned · D needs 51; R needs 50 Democratic control 43.5% Republican control 53.2% Control depends on alliances 3.3%
        - paragraph: King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
        - paragraph: A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.
        - text: "D caucus: 50.1 expected seats (46–55, modeled 90% range) R caucus: 49.6 expected seats (44–54, modeled 90% range) Unaligned: 0.3 expected seats (0–2, modeled 90% range)"
        - paragraph: Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.
      - region "What has been tested?":
        - heading "What has been tested?" [level=2]
        - paragraph: The 2026 election has not happened yet. Select 2018 or 2022 to see how historical reconstructions compare with actual results. Those views use a different model of possible winners, so their scores do not measure the accuracy of this 2026 two-contender forecast.
        - group: Scope and limitations
      - region "The races":
        - heading "The races" [level=2]
        - text: 10 of 35 elections
        - paragraph: The favorite is the candidate or group with the highest modeled chance of winning. “Most competitive” puts the least certain favorites first.
        - textbox "Search states or candidates":
          - /placeholder: Find a state or candidate
        - combobox "Sort races":
          - option "Most competitive" [selected]
          - option "State A–Z"
        - table:
          - rowgroup:
            - row "State / election Favorite Chance to win Outlook":
              - columnheader "State / election"
              - columnheader "Favorite"
              - columnheader "Chance to win"
              - columnheader "Outlook"
          - rowgroup:
            - row "Ohio Special election Jon Husted Republican 52.6% Tossup":
              - cell "Ohio Special election":
                - button "Ohio Special election"
              - cell "Jon Husted Republican"
              - cell "52.6%"
              - cell "Tossup"
            - row "Texas Regular election KEN PAXTON Republican 58.6% Tossup":
              - cell "Texas Regular election":
                - button "Texas Regular election" [pressed]
              - cell "KEN PAXTON Republican"
              - cell "58.6%"
              - cell "Tossup"
            - row "Maine Regular election Jackson, Troy D. Democratic 63.0% Likely":
              - cell "Maine Regular election":
                - button "Maine Regular election"
              - cell "Jackson, Troy D. Democratic"
              - cell "63.0%"
              - cell "Likely"
            - row "Alaska Regular election Sullivan, Dan S. Republican 64.4% Likely":
              - cell "Alaska Regular election":
                - button "Alaska Regular election"
              - cell "Sullivan, Dan S. Republican"
              - cell "64.4%"
              - cell "Likely"
            - row "Iowa Regular election Ashley Hinson Republican 68.6% Likely":
              - cell "Iowa Regular election":
                - button "Iowa Regular election"
              - cell "Ashley Hinson Republican"
              - cell "68.6%"
              - cell "Likely"
            - row "Michigan Regular election Abdul El-Sayed Democratic 68.8% Likely":
              - cell "Michigan Regular election":
                - button "Michigan Regular election"
              - cell "Abdul El-Sayed Democratic"
              - cell "68.8%"
              - cell "Likely"
            - row "Kansas Regular election Roger Marshall Republican 74.5% Likely":
              - cell "Kansas Regular election":
                - button "Kansas Regular election"
              - cell "Roger Marshall Republican"
              - cell "74.5%"
              - cell "Likely"
            - row "Minnesota Regular election Peggy Flanagan Democratic 80.3% Likely":
              - cell "Minnesota Regular election":
                - button "Minnesota Regular election"
              - cell "Peggy Flanagan Democratic"
              - cell "80.3%"
              - cell "Likely"
            - row "South Carolina Regular election Darline Graham Republican 82.7% Likely":
              - cell "South Carolina Regular election":
                - button "South Carolina Regular election"
              - cell "Darline Graham Republican"
              - cell "82.7%"
              - cell "Likely"
            - row "Nebraska Regular election Pete Ricketts Republican 84.4% Likely":
              - cell "Nebraska Regular election":
                - button "Nebraska Regular election"
              - cell "Pete Ricketts Republican"
              - cell "84.4%"
              - cell "Likely"
        - button "See all 35 races"
      - complementary "Selected race detail":
        - text: Race in focus Tossup
        - heading "Texas" [level=2]
        - text: U.S. Senate · Regular election
        - separator
        - heading "Chance of winning" [level=3]
        - paragraph: A 70% chance means winning about 70 out of 100 elections simulated by this model. It does not mean winning 70% of the votes.
        - text: JAMES TALARICO Democratic
        - strong: 41.4%
        - text: KEN PAXTON Republican
        - strong: 58.6%
        - separator
        - heading "Vote share among these two contenders" [level=3]
        - text: TALARICO
        - strong: 49.4%
        - paragraph: "Modeled 90% range: 45.2%–53.7% Share of votes for the two named principal contenders"
        - text: PAXTON
        - strong: 50.6%
        - paragraph: "Modeled 90% range: 46.3%–54.8% Share of votes for the two named principal contenders"
        - paragraph: The shares divide votes between these two contenders only; votes for other candidates are excluded. The range runs from the 5th to the 95th percentile of simulated shares—the middle 90%. It is a model estimate, not a poll’s margin of error or a guarantee.
        - paragraph: Probabilities of 0% or 100% reflect the saved simulation, not certainty.
        - separator
        - text: Selected model setup
        - strong: Standard polling
        - text: Poll records included
        - strong: "17"
        - paragraph:
          - text: This counts poll records used for this race, not people interviewed or every poll available.
          - link "Why some polls are included—and others are not ↓":
            - /url: "#poll-selection"
        - group: Race assumptions
        - separator
        - heading "Candidate and result sources" [level=3]
        - link "JAMES TALARICO ↗":
          - /url: https://www.sos.state.tx.us/elections/forms/2026-ballot-cert.pdf
        - link "KEN PAXTON ↗":
          - /url: https://www.sos.state.tx.us/elections/forms/2026-ballot-cert.pdf
        - paragraph: "Forecast date: October 26, 2026 Inputs dated through: October 26, 2026 Saved research forecast; new polls are not added automatically."
    - region "How this forecast works.":
      - text: From evidence to possible elections
      - heading "How this forecast works." [level=2]
      - paragraph: A model is a set of mathematical assumptions connecting evidence to possible results. Your choices change those assumptions and, when available, load the matching saved forecast.
      - link "Download this forecast’s data ↓":
        - /url: /data/explorer/current-polling.json
      - paragraph: "The download is a JSON file: structured data for analysis, with source identifiers and limitations."
      - heading "1. Start with a baseline" [level=3]
      - paragraph: Electoral history supplies a starting estimate, often called “fundamentals.” The builder also offers a historical winner reference model. A component is one part of the model; the chosen combination is your model setup or “stack.”
      - heading "2. Add the selected evidence" [level=3]
      - paragraph:
        - text: This setup adds the polling evidence saved for this election. Polls measure responses at a particular time. The model allows for measurement errors and changes before election day.
        - link "Read the poll selection rules below.":
          - /url: "#poll-selection"
      - heading "3. Simulate elections together" [level=3]
      - paragraph: A simulation is one possible set of election results generated by the model. Shared uncertainty allows many races to move together—for example, if polls miss in the same direction. Win chances and chamber control summarize these simulated elections, rather than assuming each race is unrelated.
      - heading "4. Explore what changes" [level=3]
      - paragraph: Changing a choice loads a saved result if one exists. Some combinations require calculating several model parts together again (a “joint refit”); the website explains when that work is needed. A sensitivity test changes assumptions without adding evidence. “Runs successfully,” “experimental,” and “tested against election results” describe different things.
      - heading "What these numbers cover" [level=3]
      - paragraph: The 2026 preview models only the two named contenders in each race. Its vote shares divide votes between those two; other candidates’ shares and chances of winning are outside this model.
      - heading "Make fair comparisons" [level=3]
      - paragraph: Compare the same year, possible winners and assumptions about independent senators. A change from 40% to 45% is 5 percentage points (“pp”), not a 5% relative increase. Historical scores describe those elections and do not establish the accuracy of the current forecast.
    - region "Why some polls are included—and others are not":
      - text: The evidence behind this view
      - heading "Why some polls are included—and others are not" [level=2]
      - paragraph:
        - text: This saved forecast uses
        - strong: 131 poll records across 25 of 35 races
        - text: ", from sources collected by October 26, 2026. A record is one selected survey question for one race. It is not a count of people interviewed, and this collection does not include every published poll."
      - heading "Match the contest and the date" [level=3]
      - paragraph: The race and named contenders must match the forecast. Sources must document usable percentages and show that the evidence was available by the saved cutoff. Fieldwork must end within 100 days of election day. An availability check does not establish the exact day a poll was published.
      - heading "Avoid counting the same evidence twice" [level=3]
      - paragraph: Only one question is chosen from a survey, so alternative ballot questions do not become extra polls. For each race, the latest eligible sample in each polling-provider grouping is retained. Where the actual survey contractor is unknown, the named survey organization supplies that grouping; this is an approximation, not a verified contractor identity.
      - heading "Keep source gaps visible" [level=3]
      - paragraph: Missing or uncertain source documents, dates or suitable questions can keep a record out. An absent poll may simply be outside the collected sources. Its absence is not, by itself, a judgment that the pollster is unreliable.
      - group: How sample sizes, voter types and missing polls are handled
      - group:
        - text: Check included records and recorded exclusions by state
        - paragraph: This list covers the reviewed source collection, including alternative questions. It is not a list of all polls conducted. “Not included” gives the recorded reason; it does not score a pollster’s quality.
        - paragraph: “Dependent” questions use the same survey respondents. A head-to-head question compares two candidates; a multicandidate question lists more. In ranked-choice voting (RCV), lower-ranked candidates are eliminated and ballots transfer to the next remaining choice. A proxy date is an estimated substitute for an unverified publication date.
        - text: Poll records for
        - combobox "Poll records for":
          - option "All states"
          - option "Alabama"
          - option "Alaska" [selected]
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
          - option "Ohio"
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
        - list:
          - listitem:
            - strong: co-efficient
            - text: · Alaska Not included
            - paragraph: "Interview dates: 2026-09-14 to 2026-09-17. Recorded publication date: 2026-09-17. Likely voters. First-choice ballot question"
            - paragraph: This question measures first choice or a head-to-head matchup, not the final ranked-choice preference used by this model.
            - paragraph: "Sample size for this exact question is not verified. Reported overall survey sample: 799 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://battlegrounds.coefficient.org/ak/
          - listitem:
            - strong: co-efficient
            - text: · Alaska Not included
            - paragraph: "Interview dates: 2026-09-14 to 2026-09-17. Recorded publication date: 2026-09-17. Likely voters. Head-to-head ballot question"
            - paragraph: This question measures first choice or a head-to-head matchup, not the final ranked-choice preference used by this model.
            - paragraph: "Sample size for this exact question is not verified. Reported overall survey sample: 799 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://battlegrounds.coefficient.org/ak/
          - listitem:
            - strong: Trafalgar Group
            - text: · Alaska Not included
            - paragraph: "Interview dates: 2026-09-22 to 2026-09-24. Exact publication date not verified. Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date). Likely voters. Ranked-choice first preference only"
            - paragraph: This question measures first choice, not the final ranked-choice preference used by this model.
            - paragraph: "Reported sample size for this question: 1079. Reported overall survey sample: 1079 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://www.thetrafalgargroup.org/wp-content/uploads/2026/09/AK-Midterms-Survey-Report-0925.pdf
          - listitem:
            - strong: InsiderAdvantage
            - text: · Alaska Included
            - paragraph: "Interview dates: 2026-10-09 to 2026-10-10. Recorded publication date: 2026-10-11. Likely voters. Ranked-choice final preference"
            - paragraph: Selected question from the latest retained sample for this race and polling-provider grouping.
            - paragraph: "Sample size for this exact question is not verified. Reported overall survey sample: 1200 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://insideradvantage.com/insideradvantage-u-s-senate-surveys-alaska-ranks-choice-peltola-50-sullivan-48-kansas-hamilton-48-marshall-47-michigan-el-sayed-48-rogers-46/
          - listitem:
            - strong: Data for Progress
            - text: · Alaska Not included
            - paragraph: "Interview dates: 2026-10-08 to 2026-10-16. Exact publication date not verified. Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date). Likely voters. Ranked-choice first preference"
            - paragraph: This survey also reports final ranked-choice preferences, which were selected for this model. Its first-choice, earlier-round and hypothetical alternatives are not added as separate polls.
            - paragraph: "Preferred question in this same survey: Ranked-choice final preference; Likely voters (included)."
            - paragraph: "Reported sample size for this question: 694. Reported overall survey sample: 694 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://www.filesforprogress.org/datasets/2026/10/dfp_alaska_oct_2026.pdf
          - listitem:
            - strong: Data for Progress
            - text: · Alaska Not included
            - paragraph: "Interview dates: 2026-10-08 to 2026-10-16. Exact publication date not verified. Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date). Likely voters. Ranked-choice question excluding unsure respondents"
            - paragraph: This survey also reports final ranked-choice preferences, which were selected for this model. Its first-choice, earlier-round and hypothetical alternatives are not added as separate polls.
            - paragraph: "Preferred question in this same survey: Ranked-choice final preference; Likely voters (included)."
            - paragraph: "Reported sample size for this question: 675. Reported overall survey sample: 694 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://www.filesforprogress.org/datasets/2026/10/dfp_alaska_oct_2026.pdf
          - listitem:
            - strong: Data for Progress
            - text: · Alaska Included
            - paragraph: "Interview dates: 2026-10-08 to 2026-10-16. Exact publication date not verified. Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date). Likely voters. Ranked-choice final preference"
            - paragraph: Selected question from the latest retained sample for this race and polling-provider grouping.
            - paragraph: "Reported sample size for this question: 671. Reported overall survey sample: 694 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://www.filesforprogress.org/datasets/2026/10/dfp_alaska_oct_2026.pdf
          - listitem:
            - strong: Data for Progress
            - text: · Alaska Not included
            - paragraph: "Interview dates: 2026-10-08 to 2026-10-16. Exact publication date not verified. Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date). Likely voters. Hypothetical ranked-choice result that reassigns exhausted ballots"
            - paragraph: This survey also reports final ranked-choice preferences, which were selected for this model. Its first-choice, earlier-round and hypothetical alternatives are not added as separate polls.
            - paragraph: "Preferred question in this same survey: Ranked-choice final preference; Likely voters (included)."
            - paragraph: "Reported sample size for this question: 675. Reported overall survey sample: 694 · Likely voters. These are publisher sample labels, not model weights or reconstructed vote counts."
            - paragraph: The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.
            - link "Publisher or source record ↗":
              - /url: https://www.filesforprogress.org/datasets/2026/10/dfp_alaska_oct_2026.pdf
    - text: ballotmancy Independent research. Visible assumptions. 2026 Senate research preview · October 26, 2026
- alert
```

# Test source

```ts
  1  | import { expect, test } from "@playwright/test";
  2  | 
  3  | test("refreshed poll explanations connect counts, date bounds, sample Ns and provider proxies", async ({ page }) => {
  4  |   await page.goto("/?run=current-polling&race=2026%3ATX%3Asenate%3Aregular");
  5  |   const detail = page.getByRole("complementary", { name: "Selected race detail" });
  6  |   await expect(detail.getByText("Poll records included", { exact: true })).toBeVisible();
  7  |   await detail.getByRole("link", { name: /Why some polls are included/ }).click();
  8  |   const policy = page.locator("#poll-selection");
  9  |   await expect(policy).toContainText("131 poll records across 25 of 35 races");
  10 |   const disclosure = policy.locator(".poll-records > summary");
  11 |   await disclosure.focus();
  12 |   await page.keyboard.press("Enter");
  13 |   await expect(page.getByLabel("Poll records for", { exact: true })).toHaveValue("TX");
  14 |   await expect(policy.locator(".poll-record-status.included")).toHaveCount(17);
  15 |   await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(7);
  16 |   await expect(policy.locator(".poll-record-list")).toContainText("YouGov");
  17 |   await expect(policy.locator(".poll-record-list a").first()).toHaveAttribute("href", /^https?:\/\//);
  18 |   await page.getByLabel("Poll records for", { exact: true }).selectOption("SD");
  19 |   await expect(policy.locator(".poll-record-status.included")).toHaveCount(4);
  20 |   await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(4);
  21 |   await page.getByLabel("Poll records for", { exact: true }).selectOption("AK");
  22 |   await expect(policy.locator(".poll-record-status.included")).toHaveCount(2);
  23 |   await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(6);
  24 |   const akQuestion = policy.locator(".poll-record-list li")
  25 |     .filter({ hasText: "Data for Progress" })
  26 |     .filter({ has: policy.locator(".poll-record-status.included") });
> 27 |   await expect(akQuestion).toContainText("Exact publication date not verified.");
     |                            ^ Error: expect(locator).toContainText(expected) failed
  28 |   await expect(akQuestion).toContainText("Source confirmed available by Mon, 26 Oct 2026 15:03:57 GMT (not a publication date).");
  29 |   await expect(akQuestion).toContainText("Reported sample size for this question: 671.");
  30 |   await expect(akQuestion).toContainText("Reported overall survey sample: 694 · Likely voters.");
  31 |   await expect(akQuestion).toContainText("polling-provider grouping uses the named survey organization");
  32 |   await expect(akQuestion.getByRole("link", { name: "Publisher or source record ↗" })).toHaveAttribute("href", /^https:\/\//);
  33 |   const alternative = policy.locator(".poll-record-list li").filter({ hasText: /Ranked-choice first preference/ });
  34 |   await expect(alternative).toContainText("final ranked-choice preferences");
  35 |   await expect(alternative).toContainText("Preferred question in this same survey: Ranked-choice final preference; Likely voters (included).");
  36 | 
  37 |   await policy.locator(".assumptions > summary").click();
  38 |   await expect(policy).toContainText("22 included records have no verified exact publication date.");
  39 |   await expect(policy).toContainText("that time is not presented as their release date");
  40 |   await page.getByLabel("Poll records for", { exact: true }).selectOption("CO");
  41 |   await expect(policy).toContainText("That does not establish that no polls were conducted.");
  42 |   await page.getByLabel("Poll records for", { exact: true }).selectOption("");
  43 |   await expect(policy.locator(".poll-record-status.included")).toHaveCount(131);
  44 |   await expect(policy.locator(".poll-record-status.excluded")).toHaveCount(56);
  45 |   expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  46 | });
  47 | 
  48 | test("poll source explanation follows the selected model and historical evidence limits", async ({ page }) => {
  49 |   await page.goto("/?run=current-fundamentals");
  50 |   const policy = page.locator("#poll-selection");
  51 |   await expect(policy).toContainText("No polls are included.");
  52 |   await expect(policy.locator(".poll-records")).toHaveCount(0);
  53 |   await page.locator(".stack-builder > summary").click();
  54 |   await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).check();
  55 |   await expect(policy).toContainText("131 poll records across 25 of 35 races");
  56 |   await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2018", exact: true }).click();
  57 |   await expect(policy).toContainText("cutoff 14 days before the election");
  58 |   await expect(policy.locator(".poll-records")).toHaveCount(0);
  59 |   const scoreboard = page.getByRole("region", { name: "How this model did", exact: true });
  60 |   await expect(scoreboard).toContainText("Recreated in 2026");
  61 |   const quality = scoreboard.getByRole("article", { name: "Probability quality", exact: true });
  62 |   await expect(quality.locator(".scoreboard-value")).toHaveText(/^0\.\d{3}$/);
  63 |   await expect(quality.locator(".scoreboard-raw")).not.toHaveAttribute("open", "");
  64 |   await expect(quality).toContainText("0 is perfect on this 0–2 scale");
  65 | });
  66 | 
```