const header = document.querySelector("[data-header]");
const heroSlides = Array.from(document.querySelectorAll("[data-hero-slide]"));
const heroDots = Array.from(document.querySelectorAll("[data-hero-dot]"));
const prevButton = document.querySelector("[data-carousel-prev]");
const nextButton = document.querySelector("[data-carousel-next]");
let activeSlideIndex = 0;
let carouselTimer;

const updateHeader = () => {
  if (!header) return;
  header.classList.toggle("is-scrolled", window.scrollY > 8);
};

const setHeroSlide = (index) => {
  if (!heroSlides.length) return;
  activeSlideIndex = (index + heroSlides.length) % heroSlides.length;

  heroSlides.forEach((slide, slideIndex) => {
    slide.classList.toggle("is-active", slideIndex === activeSlideIndex);
  });

  heroDots.forEach((dot, dotIndex) => {
    dot.classList.toggle("is-active", dotIndex === activeSlideIndex);
  });
};

const restartCarousel = () => {
  if (!heroSlides.length || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  window.clearInterval(carouselTimer);
  carouselTimer = window.setInterval(() => setHeroSlide(activeSlideIndex + 1), 7000);
};

prevButton?.addEventListener("click", () => {
  setHeroSlide(activeSlideIndex - 1);
  restartCarousel();
});

nextButton?.addEventListener("click", () => {
  setHeroSlide(activeSlideIndex + 1);
  restartCarousel();
});

heroDots.forEach((dot, index) => {
  dot.addEventListener("click", () => {
    setHeroSlide(index);
    restartCarousel();
  });
});

window.addEventListener("scroll", updateHeader, { passive: true });
updateHeader();
setHeroSlide(0);
restartCarousel();
