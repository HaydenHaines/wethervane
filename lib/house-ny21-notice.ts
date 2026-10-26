/** Website-owned notice for the exact reviewed October 25, 2026 House result bytes.
 * Owner erratum SHA-256: 9159c561ca3ac223d5bc04313c87e1de4b542d1de4ea4299d95a3c3911ee2e44.
 * Keep this separate from native artifacts; a replacement result hash needs a new review.
 */
import type { HouseLiveView } from "./house-live-view";

export const houseNy21ReviewedResults: Readonly<Record<string, string>> = {
  "house-current-fundamentals": "4cc3d17c5a39f24292f0fdee039f9f7dab5b96e3614b6d7e35df7f9459046ce4",
  "house-current-polling": "c84248d56ff0fd0e291538729680cf8cdaa51bad9fb456c3d359fa022b76451d",
  "house-current-comparison-recent_calendar_history-f": "9f98bb094bbf23c0c9bb840bbce6986dd5acaee8a777b8054fc9b994c80d4a71",
  "house-current-comparison-recent_calendar_history-fp": "7e485686194aadbb244145701f5da99597f753fda424be7c77ca5a73ccc5471a",
  "house-current-comparison-house_history_refreshed_v1-f": "e8d0bcc8584ffd821273b332d764fe81bcd3463ec360c00442be40a152bfbb61",
  "house-current-comparison-house_history_refreshed_v1-fp": "937fdf46979e12a54151a563f99aaf7964b0b361c5d51d9d0eabe8adef01e3b5",
  "house-current-comparison-house_presidential_slope_v1-f": "431bbde1da4c48ebc2e17ced53283f86425e1f75b834ebbd42f558c83b61116f",
  "house-current-comparison-house_presidential_slope_v1-fp": "11e2549492c5c35652b359bb4e0e9cc70cda5e38cc0d00ae612dbc196d50d539",
  "house-current-comparison-house_relative_lean_v1-f": "6a65d316c58b2c88208982f01fa383b7f72d1917d5de765c263a2067e04c04d8",
  "house-current-comparison-house_relative_lean_v1-fp": "4523b29273d28c0892b7b99ad65c3dada7de986137df762b431db8bf374fc98b",
  "house-current-comparison-house_relative_context_v1-f": "6b42164b32465906ae3eed90c7b8226743bc1bb9b8c7c5caee3e055aa0190025",
  "house-current-comparison-house_relative_context_v1-fp": "0bc899399caae440a11d618e181b396853e393aed2a8c40d9f9ca9c4007bee6f",
  "house-current-cycle-low": "a167aeb373826027065b9d3a4736ec635f805c38dbbea773c868ebe5f32c2efb",
  "house-current-cycle-high": "1baa27367cda0dc38d2ad5145ac4fe174580d55c4fd27af2f7ac4fec27ba5bdc",
  "house-current-error-low": "b992c9ac9c28396cd27b6e695967e23babe30091b3d7762a0d18c71f754de01b",
  "house-current-error-high": "19975e01180dd3ffc6abbc0c392ce6300a940cd6caf659c388c8aba347a48f7f",
};

export const houseNy21Notice = {
  raceId: "2026:NY-21:house:regular",
  districtKey: "NY-21",
  issuedDate: "2026-10-26",
  publicWording: "NY-21 is under review. The October 25 forecast omits declared write-in candidate Christopher Schmidt and excludes two polls that name him. These published estimates have not yet been updated for that finding.",
  explanation: "We are reviewing the candidate list and how to interpret polls that name write-in candidates. The displayed probabilities remain the October 25 estimates.",
  sources: [
    { label: "New York candidate list", url: "https://elections.ny.gov/certification-november-3-2026-general-election", date: "2026-10-07" },
    { label: "NCPR: write-in campaign", url: "https://www.northcountrypublicradio.org/news/story/54097/20261016/ny-21-christopher-schmidt-recently-polled-at-5-will-that-impact-the-race", date: "2026-10-16" },
    { label: "Siena poll", url: "https://sri.siena.edu/2026/10/09/times-union-wptz-nbc5-wamc-siena-poll-of-likely-ny-21-voters/", date: "2026-10-09" },
  ],
} as const;

export function houseNy21NoticeFor(view: HouseLiveView): typeof houseNy21Notice | null {
  return houseNy21ReviewedResults[view.run.id] === view.run.resultSha256 &&
    view.races.some(race => race.id === houseNy21Notice.raceId && race.district.districtKey === houseNy21Notice.districtKey)
    ? houseNy21Notice : null;
}
