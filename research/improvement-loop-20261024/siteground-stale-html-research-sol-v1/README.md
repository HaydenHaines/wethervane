# SiteGround stale HTML: bounded read-only investigation

Cause remains undetermined. Local deploy checkout is317ff150f012871aec6c71ae680d7e76eec5b4db. No build, hosting change, cache purge, redeploy, message to provider or site edit occurred. No SiteGround connector was discovered. Public responses and local source pins are retained separately from documentation/inference.

## Captured evidence

At01:14:51UTC2026-10-27, GEThttps://ballotmancy.com/ returned old metadata, nginx,x-proxy-cache:HIT,x-nextjs-cache:HIT,x-nextjs-prerender:1 and s-maxage=31536000. The6313-byte body hash wasddc5cb172516a1b49df1bff99f02f8cb0674d412969703556715ee719e6082e2. SiteGround documents proxyHIT as a cached response, but that observation alone cannot identify whether the active app/build is old.

Root reported fresh-query MISS observations at01:15:08UTC; its exact URLs/report are preserved in unique-get-observations.json as external observations without claiming independently retained original headers. I independently repeated those URLs at01:15:42UTC; both were now proxyHIT with identical old body/hash. Then independently requested new exact URLs at01:15:53UTC:

- https://ballotmancy.com/?site_check=317ff15-sol-20261027T0116
- https://haydenh6.sg-host.com/?site_check=317ff15-sol-20261027T0116

Both returned proxyMISS,NextHIT,prerender1,s-maxage31536000,etag11evpowq77a4v7 and the identical old6313-byte body/hash. Complete headers and HTML are retained, not just parsed snippets. The description still says the House perspective is in preparation. A stale SiteGround proxyHIT alone therefore cannot explain these fresh responses. An old active build, framework prerender/cache or another upstream layer remains possible; proxyMISS does not prove an uncached render from the intended new commit. Shared custom/temporary-host behavior narrows a custom-domain-only explanation but is not proof of architecture or cause.

## Official documentation and interpretations

[Next.js CDN caching](https://nextjs.org/docs/app/guides/cdn-caching) documents one-year s-maxage for static pages without revalidation. That header is therefore compatible with normal prerender behavior; it is not evidence of a misconfiguration by itself. Browser-independent curl reproduces old HTML, ruling out a browser-only cache explanation for these requests. Clearing a browser cannot update stale server/edge output. Hashed/_next/static assets have distinct immutable caching semantics; new assets alone would not prove a new HTML document. The guide also distinguishes CDN cache invalidation from framework revalidation.

[Next.js self-hosting](https://nextjs.org/docs/app/guides/self-hosting) describes build IDs and consistent build/runtime deployment. A local successful build proves neither remote build success nor activation of that output. Active older build artifacts, an old process, mixed instances or persisted route output remain hypotheses requiring deployment-side evidence, not diagnoses. Current docs are v16.4.3; local dependency is16.4.2, so this is documented architecture guidance rather than proof of every patch-version behavior.

[SiteGround Node.js Projects](https://www.siteground.com/blog/hosting-node-js-projects-with-siteground) documents linked-branch push-triggered rebuild/deploy, pauseable auto-deploy, and deployment records with branch,commit,build output. That requires the correct linked repository/branch and enabled integration. Its staff clarification locates build logs in Site Tools→Node.js→Logs and runtime errors under Statistics→Error Log; available access varies by product/account. The inaccessible dashboard leaves actual latest successful commit/build activation unknown.

[SiteGround cache headers](https://www.siteground.com/kb/siteground-dynamic-caching-configuration) documents proxyHIT/MISS/BYPASS and manual purging in Site Tools. Those indicators characterize that cache layer, not all framework/origin state. Generic SiteGround caching features may depend on the deployed product/configuration; no enabled CDN product is assumed here. See also [CDN management](https://www.siteground.com/kb/manage-cdn) for a separate CDN purge if that feature is actually active.

The local .github/workflows/siteground-web.yml is named Validate SiteGround application and contains tests/build/browser checks, no deployment operation. Thus reported Actions run40428923282 being in progress is not itself proof SiteGround is waiting for it, nor that a successful run deploys anything. SiteGround's linked-repo integration may run separately; account settings are unknown. No CI status was independently refreshed by this task.

## Discriminating checks for the account owner

1. In the connected Node.js project's deployment history, verify repository,deploy/siteground branch, auto-deploy status, exact latest attempted and latest successful commit317ff15, build result, activation time and mapped custom/temporary domain. Save full build/error logs. A missing/failed remote deployment should be resolved there before inferring stale cache.
2. Verify actual framework/Node/build/start/output settings against the successful release. If SSH access is available, inspect the deployed build identifier/artifact metadata/logs read-only; do not assume local .next/BUILD_ID equals remote, or that managed hosting permits process tools. Check whether old/mixed app instances remain via provider evidence.
3. If remote new-commit activation is established, compare document body/hash/metadata on the project's preview/domain with exact headers, including proxy and Next indicators. A cache purge through documented controls may then distinguish a stale shared copy, but should be a deliberate owner action, not an automatic diagnosis. Purging alone cannot fix an old origin build; the fresh MISS captures already caution against that simple explanation.
4. After any separately authorized fix, verify both HTML metadata and current functional UI/data hashes. Do not equate cacheMISS, newJS filenames, push acceptance or CI green with a successfully served corrected release. Keep evidence before/after; never broaden forecast qualification merely to test deployment.

This packet provides observations and a decision path, not a confirmed root cause or remote remediation. No hosting actions were taken.
