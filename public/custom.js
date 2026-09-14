/**
 * InferOps Chainlit UI Client Enhancements
 * - Handles interactive starter scenario card clicks
 * - Handles command-bar keyboard shortcuts (Cmd+K / Ctrl+K)
 * - Ensures wide layout adjustments
 */

(function () {
  'use strict';

  // Global helper for scenario cards to fill the prompt textarea
  window.inferopsFillPrompt = function (text) {
    const input = document.querySelector('#chat-input') || document.querySelector('textarea');
    if (!input) return;

    input.value = text;
    // Dispatch React synthetic events
    const event = new Event('input', { bubbles: true });
    input.dispatchEvent(event);

    // If text ends with 'resume ', put cursor at the end
    input.focus();
    if (typeof input.setSelectionRange === 'function') {
      input.setSelectionRange(text.length, text.length);
    }
  };

  // Keyboard shortcut: Cmd+K or Ctrl+K to focus composer
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      const input = document.querySelector('#chat-input') || document.querySelector('textarea');
      if (input) {
        input.focus();
      }
    }
  });

  // Watch for dynamic DOM changes to apply full-width classes if needed
  const observer = new MutationObserver(function () {
    // Keep layout wide if Chainlit dynamically mounts nested containers
    const mainContainers = document.querySelectorAll('main, .chat-container');
    mainContainers.forEach(function (el) {
      if (!el.classList.contains('inferops-wide-ready')) {
        el.classList.add('inferops-wide-ready');
      }
    });
  });

  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
  }
})();
