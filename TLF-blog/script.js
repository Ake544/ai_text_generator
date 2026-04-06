// TLF Blog JavaScript

// Handle newsletter subscription
const subscribeBtn = document.querySelector('.subscribe-btn');

subscribeBtn.addEventListener('click', () => {
    alert('Thank you for subscribing to our newsletter!');
});

// Handle article read more button
const readMoreBtns = document.querySelectorAll('.article-card .btn');

readMoreBtns.forEach((btn) => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        alert('Article coming soon!');
    });
});
