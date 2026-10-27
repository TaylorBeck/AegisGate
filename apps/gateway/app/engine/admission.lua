-- One round trip: token bucket, L1 GET, and session-affinity GET.
-- KEYS[1] rate-limit hash, KEYS[2] L1 key, KEYS[3] affinity key or "".
-- ARGV: capacity, refill tokens per second, now (ms), cost.
local rl_key = KEYS[1]
local l1_key = KEYS[2]
local aff_key = KEYS[3]

local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local tokens = redis.call('HGET', rl_key, 'tokens')
local ts = redis.call('HGET', rl_key, 'ts')
local filled
if tokens == false or ts == false then
  filled = capacity
else
  local elapsed = now_ms - tonumber(ts)
  if elapsed < 0 then
    elapsed = 0
  end
  filled = tonumber(tokens) + (elapsed / 1000.0) * refill_per_sec
  if filled > capacity then
    filled = capacity
  end
end

local allowed = 0
if filled >= cost then
  filled = filled - cost
  allowed = 1
end
if filled < 0 then
  filled = 0
end

redis.call('HSET', rl_key, 'tokens', filled, 'ts', now_ms)
local ttl_ms = math.floor((capacity / refill_per_sec) * 1000) + 1000
if ttl_ms < 1000 then
  ttl_ms = 1000
end
redis.call('PEXPIRE', rl_key, ttl_ms)

local l1 = redis.call('GET', l1_key)
local aff = false
if aff_key ~= '' then
  aff = redis.call('GET', aff_key)
end

return {allowed, l1, aff}
