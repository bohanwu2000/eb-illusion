// Run with node --test tests/test_playground.js. No additional packages required.
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')

const html = fs.readFileSync(path.join(__dirname, '../src/calibration_playground.html'), 'utf8')
const source = (html.match(/<script id="simulation-kernel">([\s\S]*?)<\/script>/) ||
  html.match(/<script>([\s\S]*?)<\/script>/))[1].split('const ids=')[0]

function loadKernel(onYield = () => {}) {
  let seed = 92381
  const math = Object.create(Math)
  math.random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return (seed + 0.5) / 4294967296
  }
  const context = vm.createContext({
    Math: math,
    setTimeout: callback => setImmediate(() => { onYield(); callback() })
  })
  vm.runInContext(source.replace(/^Chart.defaults.*$/m, ''), context)
  return context
}

const params = {J: 7, K: 4, g2: 1, mu: 0.5, se: 1, so: 0.8, sc: 0.6,
  dl: 0.2, t2: 0.1, shape: 'gaussian', R: 1000}

// Fixed-seed fixtures captured from the original manuscript playground.
// These catch changes to the data-generating model or either interval formula.
const fixtures = {
  gaussian: [0.46263551410201714, 0.704, 0.877, 0.202],
  skewed: [0.46027389978367156, 0.711, 0.869, 0.28],
  heavy: [0.4286734854456608, 0.735, 0.867, 0.313],
  bimodal: [0.5054573480242748, 0.703, 0.84, 0.146]
}
for (const [shape, expected] of Object.entries(fixtures)) {
  test(`preserves the original ${shape} model and interval results`, () => {
    const result = loadKernel().simulate({...params, shape})
    Object.values(result).forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12))
  })
}

test('batching preserves all replicates, including a partial final batch', async () => {
  let yields = 0
  const direct = loadKernel().simulate({...params, R: 735})
  const context = loadKernel(() => { yields += 1 })
  assert.equal(typeof context.simulateBatched, 'function', 'responsive batching is available')
  const batched = await context.simulateBatched({...params, R: 735}, () => true)
  for (const key of Object.keys(direct)) assert.ok(Math.abs(batched[key] - direct[key]) < 1e-12)
  assert.ok(yields > 1, 'the browser gets multiple opportunities to handle input')
})

test('a superseded run stops at the next batch and returns no stale result', async () => {
  let current = true
  const context = loadKernel(() => { current = false })
  assert.equal(typeof context.simulateBatched, 'function', 'responsive batching is available')
  const result = await context.simulateBatched({...params, R: 6000}, () => current)
  assert.equal(result, null)
})
