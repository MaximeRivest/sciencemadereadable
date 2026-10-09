// Light or dark before the first paint (a separate file: the page's policy runs no inline script).
// The reader's choice if they made one, else the device's.
(function () {
  var t = null;
  try { t = localStorage.getItem("theme"); } catch (e) {}
  document.documentElement.dataset.theme = t || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
})();
