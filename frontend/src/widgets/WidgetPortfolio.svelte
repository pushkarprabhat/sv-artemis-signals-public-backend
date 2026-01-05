

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';

  const mode = writable<'live' | 'paper'>('live');
  const portfolio = writable<any>(null);
  const loading = writable(true);
  const error = writable('');

  async function fetchPortfolio(selected: 'live' | 'paper') {
    loading.set(true);
    error.set('');
    portfolio.set(null);
    let url = selected === 'live' ? '/api/v1/portfolio/live' : '/api/v1/portfolio/paper';
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to fetch portfolio');
      portfolio.set(await res.json());
    } catch (e) {
      error.set(e.message);
    } finally {
      loading.set(false);
    }
  }

  onMount(() => {
    fetchPortfolio('live');
  });
</script>



<div class="widget bg-base-100 shadow p-4 mb-4">
  <h2 class="font-semibold mb-2">Portfolio</h2>
  <div class="mb-4 flex gap-2">
    <button class="btn btn-sm" on:click={() => {mode.set('live'); fetchPortfolio('live');}} disabled={$mode === 'live'}>Live</button>
    <button class="btn btn-sm" on:click={() => {mode.set('paper'); fetchPortfolio('paper');}} disabled={$mode === 'paper'}>Paper</button>
  </div>
  {#if $loading}
    <div>Loading portfolio...</div>
  {:else if $error}
    <div class="text-error">Error: {$error}</div>
  {:else if $portfolio}
    {#if $mode === 'live'}
      <div class="mb-2">Total Equity: <b>₹{$portfolio.total_equity?.toLocaleString()}</b></div>
      <div class="mb-2">Total P&L: <b>₹{$portfolio.total_pnl?.toLocaleString()}</b></div>
      <div class="mb-2">Sharpe Ratio: <b>{$portfolio.sharpe_ratio}</b></div>
      <div class="mb-2">Win Rate: <b>{$portfolio.win_rate}%</b></div>
    {:else}
      <div class="mb-2">Total Capital: <b>₹{$portfolio.total_capital?.toLocaleString()}</b></div>
      <div class="mb-2">Total P&L: <b>₹{$portfolio.total_pnl?.toLocaleString()}</b></div>
      <div class="mb-2">Return %: <b>{$portfolio.total_return_pct}%</b></div>
      <div class="mb-2">Sharpe: <b>{$portfolio.sharpe}</b></div>
      <div class="mb-2">Win Rate: <b>{$portfolio.win_rate}%</b></div>
      <div class="mb-2">Days Remaining: <b>{$portfolio.days_remaining}</b></div>
    {/if}
    {#if $portfolio.message}
      <div class="text-warning">{$portfolio.message}</div>
    {/if}
    <div class="mt-2 text-xs text-gray-500">For Shivaansh & Krishaansh — this dashboard pays your fees!</div>
  {/if}
</div>
