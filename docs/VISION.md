# Vision: US Political Covariation Model

## Core Hypothesis

Political covariance follows shared community identity -- religion, class/occupation, neighborhood (who you talk to daily) -- more closely than it follows administrative geography or broad demographic bins. Communities that share these features covary politically even when separated by state borders.

This is a community-structure-first approach to political behavior. It is **not** just another election forecasting model. Election forecasting is one downstream application of a system whose primary purpose is to discover, characterize, and leverage the latent community structure that drives American political behavior.

---

## Three Nested Ambitions

The project pursues three goals in order of increasing difficulty. Each level depends on the one beneath it.

### 1. Descriptive

Detect and characterize politically meaningful community types that cross administrative borders. The question at this level is: *What are the natural groupings of American communities when defined by who people actually live among, worship with, and work alongside -- rather than by county lines or Census regions?*

### 2. Explanatory

Show that community-level covariation explains variance in political outcomes beyond what standard demographics plus geography already capture. The question here is: *Does community structure carry signal that demographics and location leave on the table?*

### 3. Predictive

Produce conditional forecasts from poll data propagated through community covariance structure. The question is: *When a poll moves in one place, what does the learned covariance structure imply for places that share its community profile but have not been polled?*

---

## Intellectual Foundations

### "Who you talk to daily" theory

Community is operationalized not through self-identification but through shared institutional space. The people you encounter at church, at work, and in your neighborhood constitute your political community. This draws on Lazarsfeld, Berelson, and Gaudet (1948), Huckfeldt and Sprague (1987), and the broader Columbia tradition of contextual influence.

### Dual-output model: vote share AND turnout

Most election models focus on vote share alone. This project models vote share and turnout jointly, enabling decomposition of election-to-election swings into three distinct mechanisms:

- **Persuasion effects**: voters changing their candidate preference.
- **Mobilization effects**: changes in who shows up to vote.
- **Composition effects**: changes in the electorate's makeup through migration, aging, and registration shifts.

This decomposition matters because the same net swing in a county can arise from very different community-level dynamics, and the correct interpretation determines what the model should propagate to similar communities.

### Voter stickiness assumption

Voters move less than polls suggest. Much of what looks like persuasion in topline polling is actually phantom swing -- differential nonresponse, likely-voter screen artifacts, and compositional churn in the polling sample. Community-level political behavior has high temporal autocorrelation. This follows Gelman et al. (2016) on phantom swings, Kalla and Broockman (2018) on the minimal persuasive effects of campaign contact, and Green and Palmquist (1990) on partisan stability.

### Communities as modeling output, not input

Community types are latent archetypes discovered from data. They are not pre-defined categories imposed by the analyst. The model should surface community types that are politically coherent -- meaning they covary together across elections -- without requiring the user to specify what those types are in advance.

### Topic model framing

Counties are treated as mixtures of community types, analogous to how documents are mixtures of topics in LDA or how pixels are mixtures of endmembers in spectral unmixing. A county that is 40% exurban evangelical, 35% college-town professional, and 25% rural agricultural behaves as a weighted combination of those archetypes. The mixing proportions are the primary object of estimation.

### Hierarchical structure

Fine-grained community types nest into blocs, which nest into mega-blocs. The number of types at each level is data-determined, not fixed a priori. This mirrors the hierarchical structure found in topic models, nested stochastic block models, and biological community ecology (HMSC). The hierarchy allows the model to operate at the resolution appropriate to the question: fine-grained for county-level analysis, coarse for national narrative.

---

## What the Model Should Reveal

The following examples illustrate the kind of distinctions the model must be capable of making. They are not exhaustive; they are litmus tests.

- **Cuban Americans in FL and Mexican Americans in TX should NOT be collapsed into "Hispanic."** These communities have different institutional contexts (religion, class structure, migration history, media ecosystems) and have diverged politically. The model should discover this separation without being told to look for it.

- **Amish and Plain communities across PA, OH, IN, and WV should covary despite crossing state lines.** Their shared religious-institutional identity produces similar political behavior (low baseline turnout, correlated surges when mobilized). State-level models miss this.

- **"Black voters" may fracture along lines the model reveals.** Urban institutional Black communities (anchored by historically Black churches, HBCUs, civic organizations) may behave differently from suburban Black communities in newer Sun Belt developments and from rural Black communities in the Mississippi Delta or coastal Carolinas. The model should surface these distinctions if they exist in the covariance structure.

- **Military base communities may behave similarly regardless of state.** Fayetteville (NC), Killeen (TX), Clarksville (TN), and Colorado Springs (CO) share demographic and institutional features that may produce political covariation invisible to state-level models.

---

## What Success Looks Like

1. **Community types correspond to recognizable communities.** When the model outputs a latent type, domain experts (political journalists, local organizers, regional scholars) should be able to look at its feature profile and say: "Yes, I know what that is."

2. **Community structure predicts political covariance beyond demographics.** In a held-out test, a model that includes community-type mixing proportions should outperform a model with the same demographic and geographic covariates but without community structure, as measured by the accuracy of predicted cross-county correlation in election swings.

3. **Polling propagated through community structure produces useful estimates.** When a poll is observed in counties dominated by a particular community type, the model should generate informative posterior updates for unpolled counties that share that type -- and these updates should outperform naive geographic interpolation.

4. **Shift narratives at the community level are informative for analysts.** The model should produce community-level swing decompositions (persuasion, mobilization, composition) that help explain *why* an election shifted the way it did, not just *that* it shifted.

---

## What Falsification Looks Like

The model's core claims are empirical. The following findings would constitute falsification:

1. **Demographics plus geography capture all useful signal.** If a model with standard ACS variables and geographic fixed effects (or spatial random effects) explains county-level covariance as well as a model augmented with community-type structure, the community-structure hypothesis adds nothing.

2. **Cross-border community structure adds negligible predictive value.** If communities defined by shared religion, occupation, and neighborhood context do not covary more strongly across state lines than would be expected from demographics alone, the "who you talk to" mechanism is not operative at the scale this model targets.

3. **Community types are too unstable to model.** If the latent types shift radically from election to election (beyond what can be explained by real realignment), the framework lacks the temporal coherence needed for forecasting.

4. **Covariance estimated from past elections fails to predict future differential swings.** If the community-covariance matrix learned from 2012-2020 does not improve prediction of differential county swings in 2024 relative to a covariance-free baseline, the historical covariance structure is not informative for the future.

---

## Tracking Assumptions

This project rests on a set of modeling assumptions that must be explicitly stated, monitored, and revisited as evidence accumulates. These are tracked in [`ASSUMPTIONS_LOG.md`](ASSUMPTIONS_LOG.md).

---

## References

All research references -- papers, datasets, repositories, and methods -- are catalogued in [`SOURCE_LOG.md`](SOURCE_LOG.md).
