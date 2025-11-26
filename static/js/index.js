window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	bulmaCarousel.attach('#results-carousel', {
	  slidesToScroll: 1,
	  slidesToShow: 1,
	  loop: true,
	  infinite: true,
	  autoplay: true,
	  autoplaySpeed: 5000
	});
  
	bulmaSlider.attach();
  });

// Scroll-triggered animations
document.addEventListener('DOMContentLoaded', function() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);

  // Observe all sections
  document.querySelectorAll('.section').forEach(section => {
    section.classList.add('fade-in');
    observer.observe(section);
  });

  // Fallback for PDF images: if a browser can't render the PDF as an image,
  // automatically swap to the corresponding .jpg version.
  const pdfImages = document.querySelectorAll('img[src$=".pdf"]');
  pdfImages.forEach(img => {
    const pdfSrc = img.getAttribute('src');
    if (!pdfSrc) return;

    const jpgSrc = pdfSrc.replace(/\.pdf(\?.*)?$/i, '.jpg$1');

    const useJpg = () => {
      if (img.getAttribute('src') !== jpgSrc) {
        img.setAttribute('src', jpgSrc);
      }
    };

    // If loading the PDF fails, fall back to JPG.
    const onError = function() {
      useJpg();
      img.removeEventListener('error', onError);
    };

    img.addEventListener('error', onError);

    // If the image has already finished loading but failed, also swap.
    if (img.complete && img.naturalWidth === 0) {
      useJpg();
    }
  });
});