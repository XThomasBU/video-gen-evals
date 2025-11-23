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
});