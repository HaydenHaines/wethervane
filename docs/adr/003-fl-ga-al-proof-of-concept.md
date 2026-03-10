# ADR-003: FL+GA+AL as Proof-of-Concept Geography

## Status
Accepted

## Context
The model needs a proof-of-concept region before scaling nationally. The ideal region should satisfy several criteria:

1. **Community diversity**: Contains many distinct community types so the model has enough variation to discover meaningful structure.
2. **Political heterogeneity**: Includes competitive, recently-flipped, and safe jurisdictions so the model faces a range of prediction challenges.
3. **Cross-border communities**: Contains communities that span state lines, testing the model's ability to detect structure that administrative boundaries miss.
4. **Data availability**: Has good coverage from all priority data sources, including accessible voter files for validation.
5. **Manageable scale**: Large enough to be meaningful but small enough for rapid iteration during development.

We considered several candidate regions:
- **FL alone** (67 counties): High community diversity but no cross-border testing and limited to one state's political dynamics.
- **Upper Midwest (WI+MI+MN+IA)**: Good political heterogeneity (Rust Belt shifts) but less community-type diversity than the Southeast.
- **FL+GA+AL** (226 counties): Strong on all criteria.
- **National**: Too large for initial development iteration speed.

## Decision
Use Florida, Georgia, and Alabama (226 total counties: 67 FL + 159 GA + 67 AL) as the proof-of-concept geography.

The region provides:

- **Community type diversity (Florida)**: Cuban-American communities (Miami-Dade, concentrated in specific precincts), Puerto Rican communities (Osceola, Orange -- distinct political behavior from Cuban), Haitian communities (Broward, Palm Beach), Northern retiree communities (Southwest FL coast, The Villages), military communities (Jacksonville/NAS, Pensacola/NAS, Tampa/MacDill), evangelical communities (North FL panhandle, rural interior), university communities (Gainesville, Tallahassee), and diverse suburban/urban professional communities (Orlando, Tampa, South FL).

- **Community type diversity (Georgia)**: Black Belt counties (rural, majority-Black, high poverty), Atlanta metropolitan suburbs (rapidly diversifying, key to 2020 flip), rural Appalachian foothills (North GA, culturally distinct from Coastal Plain), Savannah/Augusta urban cores, and agricultural Coastal Plain.

- **Community type diversity (Alabama)**: Deep South rural (very high evangelical adherence, racially polarized), Black Belt (extends from GA through central AL), Birmingham metro (urban/suburban divide), Mobile (Gulf Coast, distinct from interior), Huntsville (aerospace/military/tech -- politically unusual for Alabama).

- **Political heterogeneity**: Florida is the paradigmatic swing state with county-level margins spanning -50 to +50. Georgia flipped in 2020 (by 0.24%), providing a natural validation target. Alabama is a safe Republican state, providing an anchor and testing whether the model correctly produces low uncertainty for non-competitive states.

- **Cross-border communities**: North Florida (panhandle, rural North FL) shares community structure with South Georgia and South Alabama -- evangelical, rural, Southern. The Chattahoochee River border between GA and AL bisects the Columbus/Phenix City metro area. The FL-GA border bisects the Valdosta and Jacksonville commuting zones.

- **Data availability**: Florida voter files are public record under FL statute 97.0585 and available for a nominal fee (~$5 for the full statewide file). Georgia voter files are accessible. Alabama is more restrictive but county-level aggregate data is sufficient for MVP. All three states have good MEDSL election return coverage and standard Census/ACS data.

## Consequences
**What becomes easier:**
- Development iteration is fast with 226 counties (vs. ~3,100 nationally).
- The region's community diversity means most major community types are represented, making the proof of concept more convincing.
- Georgia's 2020 flip provides a strong out-of-sample validation test: can the model, trained on pre-2020 data, predict the flip?
- Florida's accessible voter file enables individual-level validation of community type assignments.
- Cross-border communities (North FL / South GA) provide a direct test of Assumption A003.

**What becomes more difficult:**
- Several nationally-important community types are absent or underrepresented: Rust Belt industrial (key to 2016/2020 Midwest shifts), Great Plains agricultural, Mountain West (libertarian-leaning), Pacific Coast tech/progressive, New England (no party registration cultural difference), Upper Midwest Scandinavian/German heritage communities, Native American reservations.
- The model may overfit to Southeastern community patterns that do not transfer to other regions.
- Florida's unique features (non-partisan primaries, high transplant population, extreme age distribution) may make it an atypical test case.
- Three contiguous states provide limited variation in state-level political institutions (all three have similar election administration, redistricting approaches, and political culture relative to the national range).
- Scaling from 226 to ~3,100 counties will require re-examining computational choices that work at small scale.
