import { writable } from 'svelte/store';

export const theme = writable<'light' | 'dark'>('light');

export function toggleTheme() {
  theme.update((t) => (t === 'light' ? 'dark' : 'light'));
}
