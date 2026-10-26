# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: house-live-release.spec.ts >> actual staged 2026 House release >> availability and all 16 API choices match authenticated native results
- Location: tests/site/house-live-release.spec.ts:161:7

# Error details

```
Error: expect(received).toEqual(expected) // deep equality

- Expected  - 256
+ Received  +   0

@@ -1,1249 +1,993 @@
  Array [
    Object {
-     "capture_kind": "received",
      "id": "ca-ballot",
      "label": "ca ballot",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:56.620122+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "0e514ae7b14378b1837bcea99c07e6a05ce35baa9db4c3d3a2297704dc7e6017",
      "url": "https://elections.cdn.sos.ca.gov/statewide-elections/2026-general/cert-list-candidates.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "census-cd120-index",
      "label": "census cd120 index",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:33.485998+00:00",
      "role": "official_geography_and_corrections_index",
      "sha256": "adc6ce0a73570f9602fdf7f58a174a40803a7d176eb178bff9a4ae3de4bde4a9",
      "url": "https://www.census.gov/programs-surveys/decennial-census/about/rdo/congressional-districts.120th_Congress.html",
    },
    Object {
-     "capture_kind": "received",
      "id": "current-members",
      "label": "current members",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:18:45.253946+00:00",
      "role": "official_current_member_identity",
      "sha256": "01833c6c646873b2247950e8ab9c7ecc2c57fd2bf019d1014bd64a58b496ed3d",
      "url": "https://clerk.house.gov/xml/lists/MemberData.xml",
    },
    Object {
-     "capture_kind": "received",
      "id": "current-president",
      "label": "current president",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:40:59.825913+00:00",
      "role": "current_presidential_party_context",
      "sha256": "d425852cd0ef014849be55d20f8c2ee9f43e6fac0bcfe3266e4ea64a96f0daf6",
      "url": "https://www.whitehouse.gov/administration/donald-j-trump/",
    },
    Object {
-     "capture_kind": "received",
      "id": "current-vacancies",
      "label": "current vacancies",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:18:45.260985+00:00",
      "role": "official_current_vacancies",
      "sha256": "b9a5fd352c1c74c8f740ec3857767647046539521a8bf3bc0c8b0668df10f469",
      "url": "https://clerk.house.gov/Members/ViewVacancies",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-01",
      "label": "geometry 01",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:23.786514+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "0fff80891184520f63889f24d2dbee39ae3489358399702d8c3470d55ae9a508",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_01_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-02",
      "label": "geometry 02",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:23.723406+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "eb44067aba01d4e8f4513700cea0f0cf4be84f9f2e8ff41ca138f6938b2ca495",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_02_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-04",
      "label": "geometry 04",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:23.772626+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "1a908f7fd5f69989836f64aad23af20fef38cc2bd4fc8ccd183036b4f0f01431",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_04_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-05",
      "label": "geometry 05",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:23.902773+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "d6c34a891f1d9af78e1fcfe91931d3f6939925a572815422062a13460b6ddcb1",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_05_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-06",
      "label": "geometry 06",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.157126+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "5d441c64cb33ba3f6d8dee1be2ce5f7c1efa502f245ac67c36344de2567d4269",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_06_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-08",
      "label": "geometry 08",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:23.970047+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "0df19a54f14269de3268b31ff73f01b69dc4ecd2378dc538e97faaad605bc19e",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_08_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-09",
      "label": "geometry 09",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.152616+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "3012701bae6d745c083f69599ab6a4f6c0ad3e89e0c837738895b652a4a614c3",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_09_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-10",
      "label": "geometry 10",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.168636+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "7282cf2162be72b427554530762ac4a68ed1e1396f487d32571d38e43dec4ebd",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_10_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-12",
      "label": "geometry 12",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.425250+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "7d22cba78f6553e192fca414e78f96c07d2a38e96cdd6450e39b9c3ceed9a2f4",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_12_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-13",
      "label": "geometry 13",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.402906+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "414774fe1cb78cb292816646898b59d1d43b64f3c1312015a85bc57d26b1fca9",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_13_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-15",
      "label": "geometry 15",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.300065+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "5097bacbcc3e6738f450cd26ad0d72670a66246715790ed8772772a48c230a8a",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_15_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-16",
      "label": "geometry 16",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.513696+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "f281caca4ab1b294dba300ca52b8272f79ebadb6a5b066316a4bb09a6fd8dc5e",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_16_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-17",
      "label": "geometry 17",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.610821+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "035d698668fb51d34dbbe10c8340d4b2d9f99890e9f4dedf019c7da0a36c078b",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_17_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-18",
      "label": "geometry 18",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.634252+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "441ff14b94295a061e0481648bbd6e2708e6341f1af79e7f45fb58e2574c52fe",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_18_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-19",
      "label": "geometry 19",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.705415+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "09df98632953bf96763db4064f0e18ef9b821dd8af29024b5914db9a5379c0b8",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_19_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-20",
      "label": "geometry 20",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.761583+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "62110fe0d88ffd0a283f52aaa3c408d26888af482d45ff6be4ee8b31132b0db8",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_20_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-21",
      "label": "geometry 21",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.823863+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "a46476c2ece2b436efd48591c40ad20b61a33f55b3b6d11a6b375871e438101f",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_21_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-22",
      "label": "geometry 22",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.904420+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "3e614ea60510318c5e30a62f7e33a47e672b335e060b3c8adf6de1d1299e2208",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_22_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-23",
      "label": "geometry 23",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.924787+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "830c5b3baf554894f9b42af4ac5a9500cf2cd84b6dbad3b75eb9d64ad7d407d0",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_23_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-24",
      "label": "geometry 24",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:24.990097+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "3c75f0a8ff0a74e1345615e1ef7858feffce85b2a7c7eefedd424cc4edf3be57",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_24_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-25",
      "label": "geometry 25",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.049910+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "ef0deb8cc91e6e8e031011ac0510af7bb366fdabc098a5f02224c6e7d371254b",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_25_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-26",
      "label": "geometry 26",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.087950+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "fc75ae19a91fb84811e5c7fcdb6460091bc2e73fbe8d6ea1c8df60b2a40e096d",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_26_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-27",
      "label": "geometry 27",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.168780+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "d2f27a2a36d63a146e8faebb19974ab9ec1cd1fc1529f97a9e7795b7a95cc761",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_27_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-28",
      "label": "geometry 28",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.221892+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "40a80c7df0eeaa08b795e916548f0d82fac2f990b8ddcdcaf74753ca27d8de5f",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_28_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-29",
      "label": "geometry 29",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.287665+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "72ed1b73f230903153e9b315cc1ed00b3e3efab1ee1fca66be7fc5a9b549f218",
      "url": "https://www2.census.gov/geo/tiger/TIGER2024/CD/tl_2024_29_cd119.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-30",
      "label": "geometry 30",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.328288+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "d094422db502b0a5940ed2b97a5eb8b992770e93f4e666afc83eb1b4d89300f4",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_30_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-31",
      "label": "geometry 31",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.378347+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "e2b9129b469a61589d0e91d74bf023a5cea757733a4289b69f3baef930975e1c",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_31_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-32",
      "label": "geometry 32",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.440008+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "28b76d5e696ff62c8271b190e5a3121839f8cad6bec5390d4c73b5c7bd63edc6",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_32_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-33",
      "label": "geometry 33",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.479578+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "a32e9b846da7b9176b01501375a03f18ce12dca3aa63acb11f950b64f2b2d647",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_33_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-34",
      "label": "geometry 34",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.560742+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "86bca2912cc888379dd8a8f303aef145f431d01dfe4c01f3fdb61f84ff3e4027",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_34_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-35",
      "label": "geometry 35",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.590115+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "aba1ba5bfadeb1eb584560c5d39b838a1e09309efe83269248da49540f96f1ff",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_35_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-36",
      "label": "geometry 36",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.661251+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "42f5efa13be57f0e5f00d3db00086b3bcef704bc3cb833494b0f3e7cbafc4b62",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_36_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-37",
      "label": "geometry 37",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.812534+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "13f0680937b2fb426c50b07bfd6dd473a541cee69fa39d512759dd03e787b9ef",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_37_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-38",
      "label": "geometry 38",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.808581+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "2020142c8a251cbbc2b65c0800aa5123c8b460b45d494ed7256ecfaf6259c2cb",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_38_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-39",
      "label": "geometry 39",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.881910+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "4c18b9422c82639e078dacc4df32cdfa61689b1e2f92ec5b232b4ba3a2b3421b",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_39_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-40",
      "label": "geometry 40",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:25.993847+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "13fdcdf622b9b6cce8b5174f51692790ef2aa8ab5e8269c660c52cd13405837b",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_40_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-41",
      "label": "geometry 41",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.021014+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "0a725808dc3cc2dbc549ea2f205090af4bb0f70b32413e0979829167a411d697",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_41_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-42",
      "label": "geometry 42",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.076285+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "25f36969cbf6ef2bc03ebb9a6b8be54acfd555ecc2b4d7d130dafed3ee2ef79b",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_42_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-44",
      "label": "geometry 44",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.124701+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "471bec330066bb9ecc940d08f3501f6ba90eb217a9e4036aaee0fc26edea5b3d",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_44_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-45",
      "label": "geometry 45",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.213886+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "541371371ae5f80ac2374ef1903033c71d7ec1fd22135dd2ca7de79e0366238a",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_45_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-46",
      "label": "geometry 46",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.228090+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "9a2d5708d22bf75bd78a50fdc32aded343692733d260b285f1629e2dcd383b7b",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_46_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-47",
      "label": "geometry 47",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.638729+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "a1f017ae34e3a335492ded7572ec69f3d7e193ffbc144a9269fb7e1b8803d0fe",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_47_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-48",
      "label": "geometry 48",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.851518+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "3edba357705cbd64be72db0296c8d49d1d5e1fbaa1b3c2b55556c2e633d42c7f",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_48_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-49",
      "label": "geometry 49",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.701143+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "0222fbf567b75aadcf7c12c43fadf49050ae381fb4e69ae6566c76709d480223",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_49_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-50",
      "label": "geometry 50",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.855823+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "bcebac7a410e833274b502013216a6400159124845d307ca2b29599d673747ff",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_50_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-51",
      "label": "geometry 51",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:26.980901+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "657f900a2e85c9813e5fd360b021edfa128e0531bf1ce2492feff2d82e873291",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_51_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-53",
      "label": "geometry 53",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:27.062699+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "06567cf2744bfd932aa05082c5995a2853ca6eead904db0bb2b94b384a0f1e70",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_53_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-54",
      "label": "geometry 54",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:27.048720+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "dbc34329fb3a1006c6abe51d531cab6658ec161795cced9573053650ac06538c",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_54_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-55",
      "label": "geometry 55",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:27.146480+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "d37461db92c2597a300fcc3a9fefa65434a1904104b18ec041f75f0a91b69576",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_55_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "geometry-56",
      "label": "geometry 56",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:33:27.169097+00:00",
      "role": "official_2026_election_district_geometry",
      "sha256": "3d8a9a6bf6346be86f79cc4348b5d464b50f785a71495a66c618965da1d32e8c",
      "url": "https://www2.census.gov/geo/tiger/TIGER2026/CD/tl_2026_56_cd120.zip",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-ar",
      "label": "greenpapers ar",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:43.539869+00:00",
      "role": "candidate_source_discovery",
      "sha256": "c231e4c103c729df2b8d10afb74db5deec5852b359a513674cc4584764e9324b",
      "url": "https://www.thegreenpapers.com/G26/AR",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-az",
      "label": "greenpapers az",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:42.936696+00:00",
      "role": "candidate_source_discovery",
      "sha256": "545e5c5f9d21df675947ed004af1c4410ae18017177f2f8b51f32c659f969c0b",
      "url": "https://www.thegreenpapers.com/G26/AZ",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-ga",
      "label": "greenpapers ga",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:19.782314+00:00",
      "role": "secondary_current_general_roster_crosscheck",
      "sha256": "45e311ff124491a1ee4533fe1622e1375047581566dd8949fa16d10603de4d7c",
      "url": "https://www.thegreenpapers.com/G26/GA",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-id",
      "label": "greenpapers id",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:43.501977+00:00",
      "role": "candidate_source_discovery",
      "sha256": "4720090cc00afe0db7f69d83e0b6ffdc8a5a5028421b2e494da543f729126e24",
      "url": "https://www.thegreenpapers.com/G26/ID",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-mn",
      "label": "greenpapers mn",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:43.222449+00:00",
      "role": "candidate_source_discovery",
      "sha256": "7652a1629b799719ab3d05ebc0c3f07dc748caa2f382db4839a44172c9ec88f8",
      "url": "https://www.thegreenpapers.com/G26/MN",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-oh",
      "label": "greenpapers oh",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:42.934591+00:00",
      "role": "candidate_source_discovery",
      "sha256": "be8ca10a905b4eff18e2b3e95cc3ee2a644d363e75a4d53059ae51626e58508b",
      "url": "https://www.thegreenpapers.com/G26/OH",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-or",
      "label": "greenpapers or",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:43.260195+00:00",
      "role": "candidate_source_discovery",
      "sha256": "cf5897a6f72e855884fd3fa5d7179217f9ce9eb729a81f717244420eb6a051e2",
      "url": "https://www.thegreenpapers.com/G26/OR",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-ri",
      "label": "greenpapers ri",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:43.537562+00:00",
      "role": "candidate_source_discovery",
      "sha256": "ea2e8ea6838cef4ff5d3079be8e8f481d6a116cc803f644136df56a21196c954",
      "url": "https://www.thegreenpapers.com/G26/RI",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-tn",
      "label": "greenpapers tn",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:43.028626+00:00",
      "role": "candidate_source_discovery",
      "sha256": "0ce10873ec7aa624d0c900f70ef4a90cf830909da96b64925f847b6d19048bac",
      "url": "https://www.thegreenpapers.com/G26/TN",
    },
    Object {
-     "capture_kind": "received",
      "id": "greenpapers-wi",
      "label": "greenpapers wi",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:10:19.821593+00:00",
      "role": "secondary_current_general_roster_crosscheck",
      "sha256": "9f62a901b4816fc736e2dc57a6cfdf3ae07d2acc75e8462a92106edbdcfce4f6",
      "url": "https://www.thegreenpapers.com/G26/WI",
    },
    Object {
-     "capture_kind": "received",
      "id": "national-president2024",
      "label": "national president2024",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:18:45.137860+00:00",
      "role": "official_2024_national_presidential_results",
      "sha256": "68acdee2924d771b92a05cd950dec850b462c633c05563207ac7e206116e7366",
      "url": "https://www.fec.gov/resources/cms-content/documents/2024presgeresults.xlsx",
    },
    Object {
-     "capture_kind": "received",
      "id": "national-roster-discovery",
      "label": "national roster discovery",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:33.419813+00:00",
      "role": "secondary_candidate_discovery_not_admitted_roster",
      "sha256": "20fc27ab9672e0df8f85937fd27fea4a8dc130120b92ce423debd1a7d2d57ca0",
      "url": "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections",
    },
    Object {
-     "capture_kind": "received",
      "id": "ncsl-map-changes",
      "label": "ncsl map changes",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:35:18.962494+00:00",
      "role": "current_election_rules_and_map_review",
      "sha256": "139a182d2d668b72cb81acfc6b74292903efcc3e25cff055e9425ad2ab9b8bb7",
      "url": "https://www.ncsl.org/redistricting-and-census/changing-the-maps-tracking-mid-decade-redistricting",
    },
    Object {
-     "capture_kind": "received",
      "id": "nj-ballot",
      "label": "nj ballot",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:33.807772+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "d1816b5eaf0edcd287159047da6ccbbe33e8152546e4ac2f57755fcc28b0f603",
      "url": "https://nj.gov/state/elections/assets/pdf/election-results/2026/2026-official-general-candidates-us-house-0904.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "nyt-house-discovery",
      "label": "nyt house discovery",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:56.682071+00:00",
      "role": "alternate_poll_source_discovery",
      "sha256": "a57e2a9a0e7e0de238d15e92a7db5fd7e8573aa741384b0e5df74cd8412499fc",
      "url": "https://www.nytimes.com/newsgraphics/polls/house.csv",
    },
    Object {
-     "capture_kind": "rendered_primary_source",
      "id": "official-ak-reused-rendered",
      "label": "official ak reused rendered",
-     "limitations": Array [
-       "Dated official sample ballot previously captured by Senate session; original PDF bytes unavailable. October25 direct refresh failed TLS and web tool accessibility; current secondary reconciliation required.",
-     ],
      "retrieved_at": "2026-10-08T17:07:32+00:00",
      "role": "current_roster_source",
      "sha256": "7d9209ef6cf53a004401de4900d358597f52c02be3353ccb2156f1be0cb2839b",
      "url": "https://www.elections.alaska.gov/election/2026/General/SampleBallots/Tagalog%20HD5%20JD3.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-al-1",
      "label": "official ballot al 1",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:24:38.992440+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "dbb8540ace50584ad92c20fec3fc9fef0f5f24a3d317b13d2ed4eb7d578a00a0",
      "url": "https://www.sos.alabama.gov/sites/default/files/election-2026/2026GeneralElectionStateCertificationofDemocraticCandidates.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-al-2",
      "label": "official ballot al 2",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:24:39.002967+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "a3a109642073bd075596e0d2a0d3fa5c5a692b1178f12e13532a73987c2c89c6",
      "url": "https://www.sos.alabama.gov/sites/default/files/election-2026/2026GeneralElectionStateCertificationofRepublicanCandidates.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-de-0",
      "label": "official ballot de 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:56.671052+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "a8c71700165dc3647715695ba7cc17617c5adda8edbe8754d23007559131c484",
      "url": "https://elections.delaware.gov/candidates/candidatelist/genl_fcddt_2026.html",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-de-1",
      "label": "official ballot de 1",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:56.426306+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "72d76db1a4c7189121a2b26ead63a41073f4bff151e9263bd6a07d9438d21458",
      "url": "https://elections.delaware.gov/candidates/candidatelist/genl_wcddt_2026.html",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-fl-0",
      "label": "official ballot fl 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:24:39.485871+00:00",
      "role": "official_general_candidate_status_list",
      "sha256": "93c2d34fb4d8ad2a6f86af9d68a51513c9442852ae2d9ba4e8220454c7da3e23",
      "url": "https://dos.elections.myflorida.com/candidates/canlist.asp?CountyCode=ALL&FormsButton1=RUN+QUERY&OfficeCode=ALL&OfficeGroup=FED&OrderBy=NAM&StatusCode=ALL&elecid=20261103-GEN",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-il-data",
      "label": "official ballot il data",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:48:29.751464+00:00",
      "role": "official_public_candidate_search_or_export",
      "sha256": "37ccc1d87c9324c9054a6c078187d9f63653f1789c3113d090d4d97c4cef10da",
      "url": "https://www.elections.il.gov/ElectionOperations/DownloadCandidates.aspx?ID=sejIrI%2BQmww%3D",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-in-0",
      "label": "official ballot in 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:56.683313+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "7ad57ba6ee23303736b75f90e2d39bab4b55d7bee99f30d545fc75ef6839afa8",
      "url": "https://www.in.gov/sos/elections/files/Candidate_List_Abbreviated_20260922_011251PM.xlsx",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-ks-1",
      "label": "official ballot ks 1",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:48:56.352567+00:00",
      "role": "official_public_candidate_search_or_export",
      "sha256": "9bfd6657be01e1234bdbe6f8d37f7d0c158e061ff316b2a3756262d1313b16f9",
      "url": "https://www.sos.ks.gov/elections/elections_upcoming_candidate.aspx",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-ky-0",
      "label": "official ballot ky 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:42:03.190001+00:00",
      "role": "official_general_roster_discovery",
      "sha256": "11828d9fec1c17c97d83749b9017d762f4c33b21f5c137e672ce14b482956007",
      "url": "https://web.sos.ky.gov/CandidateFilings/default.aspx?elecid=87&id=4",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-la-1",
      "label": "official ballot la 1",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:58:58.052260+00:00",
      "role": "official_general_discovery",
      "sha256": "d5d0429c94225cdb174adad33c00c6725767ab3dabfdb29bff845727d52ebf1b",
      "url": "https://voterportal.sos.la.gov/CandidateInquiry/StatewideCandidate/CandidateList",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-md-0",
      "label": "official ballot md 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:57.615182+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "c2af2098a27ee7ec951c874a9541249fba1380dce200d3e542f54e79612a4e0c",
      "url": "https://elections.maryland.gov/elections/2026/general_candidates/2026_GG_representativeincongressbydistrict_candidatelist.csv",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-mt-0",
      "label": "official ballot mt 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:52:26.037059+00:00",
      "role": "official_general_discovery",
      "sha256": "dd26c1b2527f6e6c14e0e9753a6f94222ee134616b37054b7c394635cd76625b",
      "url": "https://candidatefiling.mt.gov/candidatefiling/CandidateList.aspx?e=450002987",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-nc-0",
      "label": "official ballot nc 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:56.759811+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "378ca3dde9e97f65a921c922171aa34942bcc68582ef46e713b371e0b6caf11d",
      "url": "https://s3.amazonaws.com/dl.ncsbe.gov/Elections/2026/Candidate%20Filing/2026_general_candidate_list_by_contest_federal_and_state.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-nd-0",
      "label": "official ballot nd 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:48:30.664640+00:00",
      "role": "official_public_candidate_search_or_export",
      "sha256": "ff14a67c7e5be6465a6be0a277ce0159fa207a9743db84298e94cae172a90bde",
      "url": "https://vip.sos.nd.gov/candidatelist.aspx?eid=348",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-ne-0",
      "label": "official ballot ne 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:56.936635+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "d27a071715896b8c82b8cce8d48200b4207ba10c44f1778f56f92c4efcd7cf96",
      "url": "https://sos.nebraska.gov/sites/default/files/doc/elections/2026/Final_Statewide_General_Candidate_Filing_List_9.11.26.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-nh-0",
      "label": "official ballot nh 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:57.186393+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "9e4f505d9bf236691d6ee2618a35d32b1270efdf4f91a08f5dc17e794802b02e",
      "url": "https://mm.nh.gov/files/uploads/sos/docs/2026-ge-candidate-list.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-pa-step1",
      "label": "official ballot pa step1",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:48:34.452660+00:00",
      "role": "official_public_candidate_search_or_export",
      "sha256": "77b31ca977408200e1b960b8fc18a07971b92dc1c377d7e2f76d8c7638e01259",
      "url": "https://www.pavoterservices.pa.gov/ElectionInfo/FooterLinkReport.aspx?ID=1420",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-vt-0",
      "label": "official ballot vt 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:57.371798+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "f646b3960ea5a305ad24122a6e4023b5a93fcd14eb46400952c8e68e0037e5fc",
      "url": "https://outside.vermont.gov/dept/sos/Elections_Division/election_info_resources/candidates/2026_general_election_qualified_candidates.xlsx",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ballot-wy-0",
      "label": "official ballot wy 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:23:57.307167+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "2f5727287045d2596097ddbfc84e98984f18c518009ca61e4ef194807b6080a2",
      "url": "https://sos.wyo.gov/Elections/Docs/2026/2026_WY_General_Election_Candidates.csv",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ga-writeins-cobb",
      "label": "official ga writeins cobb",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:20:11.545098+00:00",
      "role": "current_candidate_discrepancy_review",
      "sha256": "7b9274b86ef4dd385f159b345d3b269611a0afce4122ae8ec6d9f4588adf9993",
      "url": "https://assets.cobbcounty.gov/files/2026-09/2026%20Certified%20Write-In%20Candidates_2.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-hi-page2",
      "label": "official hi page2",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:07:32.074679+00:00",
      "role": "official_public_candidate_search_or_export",
      "sha256": "9edc1c3999dc1bff3c4182562f5f94697a1950da0e27dd72275b0e006c3489ea",
      "url": "https://olvr.hawaii.gov/Controls/CandidateFiling.aspx?elid=94",
    },
    Object {
-     "capture_kind": "rendered_primary_source",
      "id": "official-ia-rendered",
      "label": "official ia rendered",
-     "limitations": Array [
-       "SHA pins rendered source extraction, not original PDF. Direct download denied.",
-     ],
      "retrieved_at": "2026-10-25T15:00:00.393692+00:00",
      "role": "current_roster_source",
      "sha256": "84d68cadc146f197e29b460872739c0e698d785563f2c03d892060669fb716d2",
      "url": "https://sos.iowa.gov/sites/default/files/2026-08/2026%20General%20-%20Candidate%20List%20Database%20-%20All%20Elections_1.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-co",
      "label": "official index co",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:34.912666+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "284c3bf43c49c712b168bae891f2a391a878ba50ca38fbb249e161cc0aaa891d",
      "url": "https://www.coloradosos.gov/pubs/elections/vote/generalCandidates.html",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-ct",
      "label": "official index ct",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:37.682639+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "75ff93df140996da3f6de8d63b620d3a516d112f162831a2e6bc4ad1d1cd6bf8",
      "url": "https://portal.ct.gov/-/media/sots/early-voting/2026-voting-guide/2026_voter_guide-2-final-combined.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-hi",
      "label": "official index hi",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:35.477078+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "c726f28d7d1585e1697e063e6128de81bbe802d832bc0ef46029841c4f6c3eaf",
      "url": "https://olvr.hawaii.gov/Controls/CandidateFiling.aspx?elid=94",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-mi",
      "label": "official index mi",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:36.623172+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "e3551674a96aebc1d295336d857a9b626ab38177823ed5357214a40b038b1938",
      "url": "https://mi-boe.entellitrak.com/etk-mi-boe-prod/page.request.do?page=page.miboePublicReport&electionType=GEN&electionYear=2026",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-mo",
      "label": "official index mo",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:36.881669+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "614e004d0f051d1d176cbad7cbb54cf9857a82f552869ed840fba5a3c291f9cb",
      "url": "https://www.sos.mo.gov/CMSImages/ElectionCandidates/2026GeneralElectionCertifiedCandidates.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-nm",
      "label": "official index nm",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:39.816655+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "36438c042011a4651d1b08b6296a7ea444e5944c0cdfdfce40eba89a3c83a251",
      "url": "https://candidateportal.servis.sos.state.nm.us/CandidateList.aspx?eid=2917&cty=99",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-sd",
      "label": "official index sd",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:38.908131+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "47c2a83643fda021ef0d328933e796ac493d4340dd16d037b1fb6df01e485619",
      "url": "https://vip.sdsos.gov/candidatelist.aspx?eid=774",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-va",
      "label": "official index va",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:39.460321+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "ed09d375f6109e5bbb5893c82ce4ee4e43ef96eee43d1b63187a635e1e093814",
      "url": "https://www.elections.virginia.gov/casting-a-ballot/candidate-list/november-3-2026-gen-elect-federal-offices/",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-index-wa",
      "label": "official index wa",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:19:40.339945+00:00",
      "role": "candidate_authority_discovery_index",
      "sha256": "af848de7650a0a87dd2b197fd73484900576b90075b69e94a26ee1155185d575",
      "url": "https://voter.votewa.gov/CandidateList.aspx?e=899",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-la-rules",
      "label": "official la rules",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:35:17.905921+00:00",
      "role": "current_election_rules_and_map_review",
      "sha256": "bdca353ee5f24c143fc1cb341158be1e51d7551931824c8bc5bb3eff0515f663",
      "url": "https://www.sos.la.gov/elections-voting/types-of-elections",
    },
    Object {
-     "capture_kind": "rendered_primary_source",
      "id": "official-ma-rendered",
      "label": "official ma rendered",
-     "limitations": Array [
-       "SHA pins the captured rendered text, not the original HTTP response or PDF bytes. Direct download was denied by host.",
-     ],
      "retrieved_at": "2026-10-25T14:55:36.955388+00:00",
      "role": "current_roster_source",
      "sha256": "289bed6f18f3c43325f6f8bf3574ec0533700e08dcfab62eb2280d0f994d381a",
      "url": "https://www.sec.state.ma.us/divisions/elections/research-and-statistics/2026-state-election-candidates.htm",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-me-ballot-0",
      "label": "official me ballot 0",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:00:37.527448+00:00",
      "role": "official_general_candidate_roster",
      "sha256": "c628e48fa6298b6862e709166adf3145f7bc4a3f60929c3e6ef5a3ea324e03ea",
      "url": "https://www1.maine.gov/sos/sites/maine.gov.sos/files/inline-files/2026%20General%20Candidate%20List%20-%20FINAL.xlsx",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-me-ballot-2",
      "label": "official me ballot 2",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:00:37.530210+00:00",
      "role": "official_general_candidate_roster",
      "sha256": "86ab6cfc5d38d4c36c907b291d737d8c0b3863e05446ede0e307fa8d15682ee9",
      "url": "https://www1.maine.gov/sos/sites/maine.gov.sos/files/inline-files/Nov26%20Declared%20Write%20in%20List%20-%20Updated%2010.15.26.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ms-index3",
      "label": "official ms index3",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:56:38.307188+00:00",
      "role": "official_general_discovery",
      "sha256": "2ec6e6f802b8c9b37a2c7396533e7e4fdab22b55ef15f15639a7ad3cfcc17889",
      "url": "https://www.sos.ms.gov/content/CandidateQualifying/default.aspx",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-nv-clark-ballot",
      "label": "official nv clark ballot",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:12:51.979345+00:00",
      "role": "official_general_candidate_roster",
      "sha256": "f9e256f97332833676ec2362fb0e80aaf16497dc568f067e616b86df52e353e6",
      "url": "https://www.clarkcountynv.gov/adobe/assets/urn:aaid:aem:1db09a57-4c1b-47e1-94d9-e7782c442571/original/as/officesup-26g.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-nv-douglas-ballot",
      "label": "official nv douglas ballot",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:12:52.164345+00:00",
      "role": "official_general_candidate_roster",
      "sha256": "65a2f8a29de5c1dab410dc413891b2049fb68f36a7a5868941d5dbb01deeffd0",
      "url": "https://cdnsm5-hosted.civiclive.com/UserFiles/Servers/Server_18102038/File/Elections/2026/website/final-douglas%20-ge26-sb-revised-0828.pdf",
    },
    Object {
-     "capture_kind": "rendered_primary_source",
      "id": "official-ny-rendered",
      "label": "official ny rendered",
-     "limitations": Array [
-       "SHA pins the captured rendered text, not the original HTTP response or PDF bytes. Direct download was denied by host.",
-     ],
      "retrieved_at": "2026-10-25T14:55:36.955990+00:00",
      "role": "current_roster_source",
      "sha256": "8d7ffb017ffe5d4d4fd8d9cc00af432ae1b0a03cdb705a10f2c688e5247e01a2",
      "url": "https://elections.ny.gov/system/files/documents/2026/10/accessible-2026-general-ballot-certification-10.7.2026.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ok-election-list",
      "label": "official ok election list",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:05:34.117953+00:00",
      "role": "official_general_discovery",
      "sha256": "4a14125fdf01fb40462424429758c0d40d892a77babd7c9fe48680a4b45955bf",
      "url": "https://hosting.okelections.gov/electionlist.html",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-sc-general-ballot",
      "label": "official sc general ballot",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:03:10.597248+00:00",
      "role": "official_public_candidate_search_or_export",
      "sha256": "968a28296ef729d87cb99e34eca647f5e0c241c47c9fa99660cc870025de0e72",
      "url": "https://vrems.scvotes.sc.gov/Candidate/CandidateSearch?electionId=22596",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ut-certification",
      "label": "official ut certification",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:11:20.243027+00:00",
      "role": "official_general_candidate_roster",
      "sha256": "1d1a009e5754194c05b70666a1c7ee200c252f87f18c1fe6f1c00f65621a0956",
      "url": "https://vote.utah.gov/wp-content/uploads/2026/09/2026-General-Election-Certification-Amended-09-21.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-ut-writeins",
      "label": "official ut writeins",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:11:20.231311+00:00",
      "role": "official_general_candidate_roster",
      "sha256": "3333f66076656c3a5dbf3c4be16af75f6fe3c07b285809afb6ed9ab9bcec93a5",
      "url": "https://vote.utah.gov/wp-content/uploads/2026/09/2026-Write-In-Certification.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-wv-barbour",
      "label": "official wv barbour",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:05:34.601869+00:00",
      "role": "official_general_discovery",
      "sha256": "ab932f3427fae3c18bd4e09cfb789349bc517bf09f9ee7ea595086fb2193c19b",
      "url": "https://apps.sos.wv.gov/elections/2026GeneralSampleBallots/SAMPLE%20-%20WV%20Barbour%20261103%20General%20-%2014inch.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "official-wv-kanawha",
      "label": "official wv kanawha",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:05:35.201226+00:00",
      "role": "official_general_discovery",
      "sha256": "50f8f2a7742fa1447a0be1f516f27a686bda5a979e0698e0f2462e1219188dbd",
      "url": "https://apps.sos.wv.gov/elections/2026GeneralSampleBallots/SAMPLE%20-%20WV%20Kanawha%20261103%20General%20-%2017inch.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "oh09-voter-guide",
      "label": "oh09 voter guide",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:26:56.353680+00:00",
      "role": "current_general_roster_discrepancy_review",
      "sha256": "8185cf39aa482b02b6d1d27fb786ba9c1cee40b74e3615c2778fb033917acc82",
      "url": "https://www.ideastream.org/u-s-representative-district-9",
    },
    Object {
-     "capture_kind": "received",
      "id": "presidential-map2026",
      "label": "presidential map2026",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:33.228825+00:00",
      "role": "publisher_reconstructed_prior_presidential_results_current_map",
      "sha256": "a42879469c4d4061f879ea63e0233d65714f7a2aff0027fee9e45b7a663f329c",
      "url": "https://doc-0c-60-sheets.googleusercontent.com/export/f3kk9gtum31ktuqh54kn7i1e14/olqks1jo02ha3790tc59c25msk/1792937550000/105250506097979753968/*/1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0?format=xlsx",
    },
    Object {
-     "capture_kind": "received",
      "id": "presidential-method-index",
      "label": "presidential method index",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:32.914527+00:00",
      "role": "publisher_map_and_methodology_description",
      "sha256": "bd4386cd8e22861bec7f2ad1d84e4d8dde82455036e8395299ce676dc8581933",
      "url": "https://www.the-downballot.com/p/the-downballots-calculations-of-presidential",
    },
    Object {
-     "capture_kind": "received",
      "id": "tn-curia-4",
      "label": "tn curia 4",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:20:11.753551+00:00",
      "role": "current_candidate_discrepancy_review",
      "sha256": "55c54132a2ec49e29659f63a4a21ab42bf1be91ba4858c46d0d2017e714f2bf7",
      "url": "https://projectcuria.com/elections/candidate/H-TN-04/H2TN06188",
    },
    Object {
-     "capture_kind": "received",
      "id": "tn-firefly-4",
      "label": "tn firefly 4",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T15:20:11.650872+00:00",
      "role": "current_candidate_discrepancy_review",
      "sha256": "42509b2d858278e00b1a204fd287d21dd980924bf37d0ffa4be2772aecd31bd0",
      "url": "https://tnfirefly.com/elections/us-house/4",
    },
    Object {
-     "capture_kind": "received",
      "id": "tx-ballot",
      "label": "tx ballot",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:32.920970+00:00",
      "role": "official_general_ballot_roster",
      "sha256": "c13ffb4ebeee389fa9818d47b31f77b4e260ea6e6f5389cb7bf0f36d9b44d87c",
      "url": "https://www.sos.state.tx.us/elections/forms/2026-ballot-cert.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "tx-writeins",
      "label": "tx writeins",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:55.997879+00:00",
      "role": "official_declared_writeins",
      "sha256": "eece7f9fa38c65eef150d62ba0a142672dc187b7feb2f90bc330f9c5a226aa24",
      "url": "https://www.sos.state.tx.us/elections/forms/2026-write-In-ballot-cert.pdf",
    },
    Object {
-     "capture_kind": "received",
      "id": "votehub-house",
      "label": "votehub house",
-     "limitations": Array [],
      "retrieved_at": "2026-10-25T14:12:56.062732+00:00",
      "role": "current_house_polls",
      "sha256": "b7a569fb017bae85c09006f6e42b85e5e05f150d900271a69e6158be5e02e90e",
      "url": "https://api.votehub.com/polls?poll_type=us-representative&subject=2026",
    },
```

# Test source

```ts
  91  |   if (!lstatSync(manifestPath).isFile() || realpathSync(manifestPath) !== manifestPath)
  92  |     throw Error("Live House publication.json must be a regular staged file.");
  93  |   const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;
  94  |   if (manifest.schema !== "HouseLiveSiteManifestV1" || manifest.status !== "available" ||
  95  |     manifest.cycle !== 2026 || manifest.results?.length !== 16 || !manifest.pollingDisclosure)
  96  |     throw Error("The real 2026 House release is absent, incomplete, or lacks polling disclosure.");
  97  |   const catalog = readPinned<Catalog>(manifest.catalog);
  98  |   const review = readPinned<Release["review"]>(manifest.review);
  99  |   const polling = readPinned<PollingDisclosure>(manifest.pollingDisclosure);
  100 |   if (catalog.schema !== "HouseLiveForecastCatalogV1" || catalog.entries?.length !== 16 ||
  101 |     catalog.default_id !== "house-current-polling" || review.decision !== "publish" ||
  102 |     review.catalog_sha256 !== manifest.catalog.sha256 || review.results?.length !== 16 ||
  103 |     polling.schema !== "HouseLiveSitePollingDisclosureV1" || polling.runs?.length !== 16)
  104 |     throw Error("The staged House catalog, review, or polling disclosure is malformed.");
  105 |   const results = new Map<string, NativeResult>();
  106 |   for (const entry of catalog.entries) {
  107 |     const staged = manifest.results.find(row => row.id === entry.id);
  108 |     const reviewed = review.results.find(row => row.id === entry.id);
  109 |     const pollRun = polling.runs.find(row => row.id === entry.id);
  110 |     if (!staged || !reviewed || staged.artifact.sha256 !== entry.result.sha256 ||
  111 |       reviewed.result_sha256 !== entry.result.sha256 || !pollRun ||
  112 |       pollRun.resultSha256 !== entry.result.sha256 || pollRun.polling !== entry.configuration.polling)
  113 |       throw Error(`The staged House release does not bind ${entry.id}.`);
  114 |     const result = readPinned<NativeResult>(entry.result);
  115 |     if (result.id !== entry.id || result.qualification.publication_status !== "reviewed" ||
  116 |       result.races?.length !== 435 || result.joint_draws?.ordered_race_ids?.length !== 435 ||
  117 |       reviewed.joint_draws_sha256 !== result.joint_draws.sha256)
  118 |       throw Error(`The staged House result is incomplete or unreviewed: ${entry.id}.`);
  119 |     results.set(entry.id, result);
  120 |   }
  121 |   if (results.size !== 16 || !results.has(catalog.default_id)) throw Error("The staged House roster is incomplete.");
  122 |   return { manifest, catalog, review, results, polling };
  123 | }
  124 | 
  125 | const native = (id: string) => {
  126 |   const result = release.results.get(id);
  127 |   if (!result) throw Error(`Missing authenticated native result ${id}.`);
  128 |   return result;
  129 | };
  130 | const configKey = (value: Configuration) => `${value.base}|${value.polling}|${value.uncertainty}`;
  131 | const probability = (value: number) => value > 0 && value < .001 ? "<0.1%"
  132 |   : value < 1 && value > .999 ? ">99.9%" : `${(value * 100).toFixed(1)}%`;
  133 | const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
  134 | const rgb = (hex: string) => `rgb(${[1, 3, 5].map(at => parseInt(hex.slice(at, at + 2), 16)).join(", ")})`;
  135 | const outlook = (result: NativeResult) => {
  136 |   const control = result.chamber?.majority;
  137 |   if (!control) return { party: "unavailable", color: "" };
  138 |   if (control.D > .6) return { party: "D", color: rgb("#3e6683") };
  139 |   if (control.R > .6) return { party: "R", color: rgb("#a3553f") };
  140 |   if (control.unresolved > .6) return { party: "U", color: rgb("#726e66") };
  141 |   return { party: "tossup", color: rgb("#7b658e") };
  142 | };
  143 | function districtColor(race: NativeRace): string {
  144 |   const leader = [...race.outcomes].sort((a, b) => b.p_win - a.p_win)[0];
  145 |   if (leader.p_win <= .6) return "#7b658e";
  146 |   const colors = leader.p_win > .9
  147 |     ? { D: "#3e6683", R: "#a3553f", O: "#726e66" }
  148 |     : { D: "#92afc1", R: "#d4a28c", O: "#beb9ad" };
  149 |   return colors[leader.party];
  150 | }
  151 | async function openBuilder(page: Page) {
  152 |   const panel = page.locator("details.house-configuration");
  153 |   if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  154 |   return panel;
  155 | }
  156 | 
  157 | test.describe("actual staged 2026 House release", () => {
  158 |   test.skip(!enabled, "Set BALLOTMANCY_LIVE_HOUSE_RELEASE=1 only after staging the reviewed real release.");
  159 |   test.beforeAll(() => { release = loadRelease(); });
  160 | 
  161 |   test("availability and all 16 API choices match authenticated native results", async ({ request }) => {
  162 |     test.setTimeout(180_000);
  163 |     const availability = await request.get("/api/chambers/house/live/availability");
  164 |     expect(availability.status()).toBe(200);
  165 |     expect(await availability.json()).toEqual({ schema: "HouseLiveAvailabilityV1", available: true });
  166 |     for (const entry of release.catalog.entries) {
  167 |       const response = await request.get(`/api/chambers/house/live?run=${encodeURIComponent(entry.id)}`);
  168 |       expect(response.status(), entry.id).toBe(200);
  169 |       const publication = await response.json();
  170 |       const result = native(entry.id);
  171 |       expect(publication.status).toBe("available");
  172 |       expect(publication.catalog.defaultId).toBe(release.catalog.default_id);
  173 |       expect(publication.catalog.entries).toHaveLength(16);
  174 |       expect(publication.view.run).toMatchObject({ id: entry.id, resultSha256: entry.result.sha256,
  175 |         configuration: entry.configuration, cycle: result.cycle, historical: false,
  176 |         issuedAt: result.issued_at, cutoffAt: result.forecast_cutoff, electionDate: result.election_date,
  177 |         strictAsOf: result.strict_as_of, sourceCutoffs: result.source_cutoffs });
  178 |       expect(configKey(publication.view.run.configuration)).toBe(configKey(result.configuration));
  179 |       expect(publication.view.races).toHaveLength(435);
  180 |       expect(publication.view.jointDraws).toMatchObject({ sha256: result.joint_draws.sha256,
  181 |         count: result.joint_draws.count, chainCount: result.joint_draws.chain_count,
  182 |         orderedRaceIds: result.joint_draws.ordered_race_ids,
  183 |         identitySemantics: result.joint_draws.identity_semantics });
  184 |       expect(publication.view.mapVintage).toBe(result.map_vintage);
  185 |       expect(publication.view.qualification).toEqual({
  186 |         inputStatus: result.qualification.input_status, numericalStatus: result.qualification.numerical_status,
  187 |         publicationStatus: result.qualification.publication_status, forecastSkill: result.qualification.forecast_skill,
  188 |         limitations: result.qualification.limitations,
  189 |         evidenceHashes: result.qualification.evidence.map(pin => pin.sha256),
  190 |       });
> 191 |       expect(publication.view.sources).toEqual(result.provenance);
      |                                        ^ Error: expect(received).toEqual(expected) // deep equality
  192 |       expect(publication.view.historicalEvidence).toEqual(result.historical_evidence);
  193 |       expect(publication.publicationReview?.limitations).toEqual(release.review.limitations);
  194 |       if (result.chamber) {
  195 |         expect(publication.view.chamber.control).toMatchObject({
  196 |           democraticMajorityProbability: result.chamber.majority.D,
  197 |           republicanMajorityProbability: result.chamber.majority.R,
  198 |           unresolvedOrganizingProbability: result.chamber.majority.unresolved,
  199 |         });
  200 |         for (const [nativeParty, displayParty] of [["D", "D"], ["R", "R"], ["O", "U"]] as const)
  201 |           expect(publication.view.chamber.seats[displayParty]).toMatchObject({
  202 |             expectedSeats: result.chamber.seats[nativeParty].expected,
  203 |             seatQuantiles: {
  204 |               p05: result.chamber.seats[nativeParty].quantiles.p05,
  205 |               p50: result.chamber.seats[nativeParty].quantiles.p50,
  206 |               p95: result.chamber.seats[nativeParty].quantiles.p95,
  207 |             },
  208 |           });
  209 |         expect(publication.view.chamberPrecision).toMatchObject({
  210 |           maxSeatMeanMcse: result.chamber.precision.max_seat_mean_mcse,
  211 |           maxControlMcse: result.chamber.precision.max_control_mcse,
  212 |         });
  213 |       } else expect(publication.view.chamber).toBeNull();
  214 |       expect(publication.view.races.map((race: { id: string }) => race.id)).toEqual(result.races.map(race => race.race_id));
  215 |       const sources = new Map(result.provenance.map(source => [source.id, source.url]));
  216 |       for (let index = 0; index < 435; index++) {
  217 |         const observed = publication.view.races[index];
  218 |         const source = result.races[index];
  219 |         const urls = (ids: string[]) => [...new Set(ids.map(id => {
  220 |           const url = sources.get(id);
  221 |           if (!url) throw Error(`Native source ID ${id} is missing from ${entry.id} provenance.`);
  222 |           return url;
  223 |         }))];
  224 |         const candidates = new Map(source.candidates.map(candidate => [candidate.candidate_id, candidate]));
  225 |         expect(observed).toMatchObject({
  226 |           id: source.race_id, state: source.state, seatId: source.district_key,
  227 |           electionLabel: "2026 House election", pollCount: source.poll_count,
  228 |           notes: source.qualifications,
  229 |           district: { cycle: result.cycle, districtKey: source.district_key, mapVintage: source.map_vintage },
  230 |           ruleFamily: source.rule_family, method: source.method,
  231 |         });
  232 |         expect(observed.candidates).toEqual(source.candidates.map(candidate => ({
  233 |           id: candidate.candidate_id, name: candidate.name, partyLabel: candidate.party_label,
  234 |           aggregationParty: candidate.aggregation_party, ballotStatus: candidate.ballot_status,
  235 |           incumbent: candidate.incumbent, sourceUrls: urls(candidate.source_ids),
  236 |         })));
  237 |         expect(observed.outcomes).toEqual(source.outcomes.map(outcome => ({
  238 |           id: `${source.race_id}:${outcome.outcome_id}`, name: outcome.label, party: outcome.party,
  239 |           kind: outcome.kind, winProbability: outcome.p_win,
  240 |           members: outcome.member_candidate_ids.map(id => {
  241 |             const candidate = candidates.get(id);
  242 |             if (!candidate) throw Error(`Native outcome member ${id} is missing from ${source.race_id}.`);
  243 |             return { id, name: candidate.name, party: candidate.aggregation_party,
  244 |               incumbent: candidate.incumbent, sourceUrls: urls(candidate.source_ids) };
  245 |           }),
  246 |         })));
  247 |         const pair = source.conditional_shares;
  248 |         expect(observed.conditionalPair).toEqual(pair ? {
  249 |           firstCandidateId: pair.first_candidate_id, secondCandidateId: pair.second_candidate_id,
  250 |           firstCandidateName: candidates.get(pair.first_candidate_id)!.name,
  251 |           secondCandidateName: candidates.get(pair.second_candidate_id)!.name,
  252 |           denominator: pair.denominator, mean: pair.mean, quantiles: pair.quantiles,
  253 |           firstExceedsHalf: pair.p_first_exceeds_half, marginWithinTwoPoints: pair.p_margin_within_2pp,
  254 |           firstMarginAtLeastTenPoints: pair.p_first_margin_at_least_10pp,
  255 |           secondMarginAtLeastTenPoints: pair.p_second_margin_at_least_10pp,
  256 |           qualifications: pair.qualifications,
  257 |         } : null);
  258 |       }
  259 |       const pollRun = release.polling.runs.find(row => row.id === entry.id);
  260 |       expect(pollRun).toMatchObject({ id: entry.id, resultSha256: entry.result.sha256,
  261 |         polling: entry.configuration.polling });
  262 |       const expectedPollQuestions = entry.configuration.polling === "standard"
  263 |         ? release.polling.selected : [];
  264 |       const nativePollIds = result.races.flatMap(race => race.poll_source_ids).sort();
  265 |       expect(expectedPollQuestions.map(row => row.observationId).sort(), entry.id).toEqual(nativePollIds);
  266 |       expect(publication.polling).toEqual({
  267 |         schema: "HouseLivePollingRunV1", runId: entry.id, resultSha256: entry.result.sha256,
  268 |         ruleset: release.polling.ruleset,
  269 |         cutoffs: release.polling.cutoffs, bindings: release.polling.bindings,
  270 |         modelUsesSelectedQuestions: entry.configuration.polling === "standard",
  271 |         usedQuestions: expectedPollQuestions,
  272 |         selectedPoolQuestionCount: release.polling.selected.length,
  273 |         selectedPoolDistrictCount: new Set(release.polling.selected.map(row => row.raceId)).size,
  274 |         excluded: release.polling.excluded, excludedReasonCounts: release.polling.excludedReasonCounts,
  275 |       });
  276 |       expect(JSON.stringify(publication)).not.toMatch(/\/home\/|file:\/\//);
  277 |     }
  278 |   });
  279 | 
  280 |   test("the real default renders its native chamber, all districts, and review notes", async ({ page }) => {
  281 |     test.setTimeout(120_000);
  282 |     const result = native(release.catalog.default_id);
  283 |     const errors: string[] = [];
  284 |     page.on("pageerror", error => errors.push(error.message));
  285 |     await page.goto("/?chamber=house");
  286 |     await expect(page.getByRole("heading", { name: /The House,\s*in perspective\./ })).toBeVisible();
  287 |     await expect(page).toHaveURL(new RegExp(`run=${release.catalog.default_id}`));
  288 |     const heading = page.locator(".hero .chamber-name");
  289 |     const expected = outlook(result);
  290 |     await expect(heading).toHaveAttribute("data-chamber-outlook", expected.party);
  291 |     if (expected.color) await expect(heading).toHaveCSS("color", expected.color);
```