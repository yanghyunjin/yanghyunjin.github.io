(() => {
  const root = document.documentElement;
  const hero = document.querySelector('.hero');
  const sticky = document.querySelector('.hero-sticky');
  const stage = document.querySelector('.device-stage');
  const heroLink = document.querySelector('.hero-link');
  const nav = document.querySelector('.product-nav');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const tall = matchMedia('(min-height: 600px)');
  const targets = [...document.querySelectorAll('.statement, .feature-copy, .feature-card, .stories .wrap, .download, .anonymous-copy, .anonymous-visual')];
  let observer;
  let frame = 0;
  const clamp = value => Math.min(1, Math.max(0, value));

  function render() {
    frame = 0;
    if (reduced.matches) return;
    const maxScroll = root.scrollHeight - innerHeight;
    root.style.setProperty('--page-progress', maxScroll > 0 ? clamp(scrollY / maxScroll) : 0);
    if (!tall.matches) return;
    const rect = hero.getBoundingClientRect();
    const available = sticky.clientHeight;
    const progress = clamp((nav.offsetHeight - rect.top) / Math.max(1, hero.offsetHeight - available));
    const endScale = Math.min(1.08, (available - 50) / stage.offsetHeight);
    const scale = endScale * (.84 + .16 * progress);
    const startTop = available * .61;
    const endTop = (available - stage.offsetHeight * endScale) / 2;
    hero.style.setProperty('--scene-y', `${startTop + (endTop - startTop) * progress}px`);
    hero.style.setProperty('--scene-scale', scale);
    hero.style.setProperty('--copy-y', `${progress * -90}px`);
    hero.style.setProperty('--copy-opacity', Math.max(0, 1 - progress * 2.3));
    hero.classList.toggle('hero-past', progress > .45);
    heroLink.tabIndex = progress > .45 ? -1 : 0;
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(render);
  }
  function configure() {
    observer?.disconnect();
    root.classList.toggle('scroll-motion', !reduced.matches);
    root.classList.toggle('pinned-hero', !reduced.matches && tall.matches);
    targets.forEach(el => el.classList.remove('reveal-pending'));
    hero.classList.remove('hero-past');
    heroLink.tabIndex = 0;
    if (!reduced.matches && 'IntersectionObserver' in window) {
      observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.remove('reveal-pending');
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: .12, rootMargin: '0px 0px -35px 0px' });
      targets.forEach(el => {
        if (el.getBoundingClientRect().top > innerHeight) {
          el.classList.add('reveal-pending');
          observer.observe(el);
        }
      });
    }
    render();
  }
  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule, { passive: true });
  addEventListener('pageshow', schedule);
  reduced.addEventListener('change', configure);
  tall.addEventListener('change', configure);
  configure();
})();
