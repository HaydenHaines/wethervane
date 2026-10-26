import { chromium, devices, expect } from '/home/hayden/projects/ballotmancy-election-night-guide-20261026/node_modules/@playwright/test/index.mjs';
import { writeFileSync } from 'node:fs';
const output='/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/election-night-result-runtime-v1';
const browser=await chromium.launch({headless:true});const captures=[];const errors=[];
try{
 for(const [name,options] of [['desktop',{viewport:{width:1280,height:900}}],['mobile',{...devices['Pixel 7']}]] ){
  const context=await browser.newContext(options),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:3137/?run=current-fundamentals');await expect(page.locator('.hero')).toBeVisible();
  await page.getByRole('button',{name:'Election night scorecard',exact:true}).click();const panel=page.locator('#election-night-scorecard');
  await expect(panel).toHaveAttribute('aria-busy','false');await expect(panel.locator('.night-primary-margin-value')).toHaveText('5.7 percentage points');
  async function capture(target,filename,kind,runId){
   await target.evaluate(element=>scrollTo({top:element.getBoundingClientRect().top+scrollY-100,behavior:'instant'}));
   await page.waitForTimeout(150);const path=`${output}/${filename}`;await page.screenshot({path,fullPage:false,animations:'disabled'});
   const info=await target.evaluate(e=>({top:e.getBoundingClientRect().top,height:e.getBoundingClientRect().height,text:e.textContent}));
   captures.push({viewport:name,kind,runId,path,viewportSize:page.viewportSize(),...info});
  }
  await capture(panel.locator('.night-primary-reference'),`${name}-primary32-fp-viewport.png`,'primary32','current-polling');
  await capture(panel.getByRole('complementary',{name:'Senate evaluation coverage'}),`${name}-senate-coverage-viewport.png`,'coverage35-core32-conditional3','current-polling');
  await panel.locator('.night-models > summary').click();await panel.getByLabel('Scorecard model').selectOption('current-fundamentals');
  await expect(panel.locator('.night-primary-margin-value')).toHaveText('9.6 percentage points');
  await capture(panel.locator('.night-primary-reference'),`${name}-primary32-f-viewport.png`,'primary32','current-fundamentals');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);await context.close();
 }
}finally{await browser.close();}
writeFileSync(`${output}/visual-capture.json`,JSON.stringify({origin:'http://127.0.0.1:3137',errors,captures},null,2)+'\n');
console.log(JSON.stringify({errors,paths:captures.map(c=>c.path)},null,2));if(errors.length)process.exitCode=1;
