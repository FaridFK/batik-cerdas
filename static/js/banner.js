document.addEventListener('DOMContentLoaded', function () {
    let currentSlide = 0;
    const slides = document.querySelectorAll('.slides img');
    const totalSlides = slides.length;

    function nextSlide() {
        currentSlide = (currentSlide + 1) % totalSlides;
        updateSlide();
    }

    function updateSlide() {
        const slideWidth = slides[0].clientWidth;
        const offset = -currentSlide * slideWidth;
        document.querySelector('.slides').style.transform = `translateX(${offset}px)`;
    }

    setInterval(nextSlide, 3000); // Ganti gambar setiap 3 detik
});
