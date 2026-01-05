<script lang="ts">
  // Layout logic: theme/layout switcher, navigation, role-based access
  import { onMount } from 'svelte';
  // ...import theme/layout stores, navigation, auth, etc.
</script>

<main class="min-h-screen bg-base-100 text-base-content">
  <!-- Topbar, Sidebar, Theme/Layout Switcher, Slot for pages -->
  <slot />
</main>
