/**
 * InferOps Chainlit UI Client Enhancements
 * - Handles interactive starter scenario card clicks
 * - Handles command-bar keyboard shortcuts (Cmd+K / Ctrl+K)
 * - Ensures single-frame composer and contained textarea layout
 */

(function () {
  'use strict';

  // Global helper for scenario cards to fill the prompt textarea
  window.inferopsFillPrompt = function (text) {
    const input = document.querySelector('#chat-input') || document.querySelector('#message-composer textarea') || document.querySelector('textarea');
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
      const input = document.querySelector('#chat-input') || document.querySelector('#message-composer textarea') || document.querySelector('textarea');
      if (input) {
        input.focus();
      }
    }
  });

  // Keep single-frame composer and layout clean on DOM mutations
  function polishComposer() {
    const composer = document.querySelector('#message-composer');
    if (composer) {
      // Ensure inner wrappers do not have borders or extra backgrounds
      const innerDivs = composer.querySelectorAll('div');
      innerDivs.forEach(function (el) {
        if (el !== composer && !el.classList.contains('inferops-clean-inner')) {
          el.classList.add('inferops-clean-inner');
        }
      });
    }

    const input = document.querySelector('#chat-input') || (composer ? composer.querySelector('textarea') : null);
    if (input && !input.classList.contains('inferops-input-ready')) {
      input.classList.add('inferops-input-ready');
      input.style.boxSizing = 'border-box';
      input.style.fontSize = '0.875rem';
      input.style.lineHeight = '1.45';
    }
  }

  const observer = new MutationObserver(function () {
    polishComposer();

    const mainContainers = document.querySelectorAll('main, .chat-container');
    mainContainers.forEach(function (el) {
      if (!el.classList.contains('inferops-wide-ready')) {
        el.classList.add('inferops-wide-ready');
      }
    });
  });

  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
    polishComposer();
  } else {
    document.addEventListener('DOMContentLoaded', function () {
      observer.observe(document.body, { childList: true, subtree: true });
      polishComposer();
    });
  }
})();
