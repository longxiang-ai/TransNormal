/**
 * TransNormal Project Page - Interactive Features
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all interactive components
    initNavbarScroll();
    initSmoothScroll();
    initComparisonSliders();
    initCopyBibtex();
});

/**
 * Navbar scroll effect - add background on scroll
 */
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    let lastScroll = 0;

    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;

        // Add shadow on scroll
        if (currentScroll > 10) {
            navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.boxShadow = 'none';
        }

        lastScroll = currentScroll;
    });
}

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const navbarHeight = document.querySelector('.navbar')?.offsetHeight || 64;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navbarHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Image comparison slider functionality
 */
function initComparisonSliders() {
    initDualComparisonSliders();
    initTripleComparisonSliders();
}

function initDualComparisonSliders() {
    const sliders = document.querySelectorAll('.comparison-container');

    sliders.forEach(slider => {
        const oursImg = slider.querySelector('.img-ours');
        const handle = slider.querySelector('.slider-handle');
        if (!oursImg || !handle) return;

        let isDragging = false;

        function updateSliderPosition(x) {
            const rect = slider.getBoundingClientRect();
            let percentage = ((x - rect.left) / rect.width) * 100;
            percentage = Math.max(0, Math.min(100, percentage));

            oursImg.style.clipPath = `inset(0 0 0 ${percentage}%)`;
            handle.style.left = `${percentage}%`;
        }

        function startDrag(e) {
            if (!e.isPrimary) return;
            if (e.pointerType === 'mouse' && e.button !== 0) return;
            isDragging = true;
            slider.setPointerCapture(e.pointerId);
            updateSliderPosition(e.clientX);
            e.preventDefault();
        }

        function onDrag(e) {
            if (!isDragging || !e.isPrimary) return;
            updateSliderPosition(e.clientX);
        }

        function stopDrag(e) {
            if (!isDragging) return;
            isDragging = false;
            if (slider.hasPointerCapture(e.pointerId)) {
                slider.releasePointerCapture(e.pointerId);
            }
        }

        slider.addEventListener('pointerdown', startDrag);
        slider.addEventListener('pointermove', onDrag);
        slider.addEventListener('pointerup', stopDrag);
        slider.addEventListener('pointercancel', stopDrag);
        slider.addEventListener('lostpointercapture', function() {
            isDragging = false;
        });
    });
}

function initTripleComparisonSliders() {
    const sliders = document.querySelectorAll('.comparison-triple-container');

    sliders.forEach(slider => {
        const inputImg = slider.querySelector('.img-input');
        const baselineImg = slider.querySelector('.img-baseline');
        const oursImg = slider.querySelector('.img-ours');
        const handleLeft = slider.querySelector('.slider-handle-left');
        const handleRight = slider.querySelector('.slider-handle-right');
        if (!inputImg || !baselineImg || !oursImg || !handleLeft || !handleRight) return;

        let isDragging = false;
        let activeHandle = 'left';
        let leftPct = 33.333;
        let rightPct = 66.666;
        const minGap = parseFloat(slider.dataset.minGap) || 10;

        function clamp(value, min, max) {
            return Math.max(min, Math.min(max, value));
        }

        function applyClips() {
            inputImg.style.clipPath = `inset(0 ${100 - leftPct}% 0 0)`;
            baselineImg.style.clipPath = `inset(0 ${100 - rightPct}% 0 ${leftPct}%)`;
            oursImg.style.clipPath = `inset(0 0 0 ${rightPct}%)`;
            handleLeft.style.left = `${leftPct}%`;
            handleRight.style.left = `${rightPct}%`;
        }

        function getPercentFromX(x) {
            const rect = slider.getBoundingClientRect();
            return clamp(((x - rect.left) / rect.width) * 100, 0, 100);
        }

        function pickActiveHandle(x) {
            const pct = getPercentFromX(x);
            const distLeft = Math.abs(pct - leftPct);
            const distRight = Math.abs(pct - rightPct);
            return distLeft <= distRight ? 'left' : 'right';
        }

        function updateHandle(x) {
            const pct = getPercentFromX(x);
            if (activeHandle === 'left') {
                let desiredLeft = pct;
                if (desiredLeft > rightPct - minGap) {
                    const pushedRight = clamp(desiredLeft + minGap, minGap, 100);
                    if (pushedRight >= 100) {
                        desiredLeft = Math.max(0, 100 - minGap);
                    }
                    rightPct = pushedRight;
                }
                leftPct = clamp(desiredLeft, 0, rightPct - minGap);
            } else {
                let desiredRight = pct;
                if (desiredRight < leftPct + minGap) {
                    const pushedLeft = clamp(desiredRight - minGap, 0, 100 - minGap);
                    if (pushedLeft <= 0) {
                        desiredRight = Math.min(100, minGap);
                    }
                    leftPct = pushedLeft;
                }
                rightPct = clamp(desiredRight, leftPct + minGap, 100);
            }
            applyClips();
        }

        function startDrag(e) {
            if (!e.isPrimary) return;
            if (e.pointerType === 'mouse' && e.button !== 0) return;
            isDragging = true;
            activeHandle = pickActiveHandle(e.clientX);
            slider.setPointerCapture(e.pointerId);
            updateHandle(e.clientX);
            e.preventDefault();
        }

        function onDrag(e) {
            if (!isDragging || !e.isPrimary) return;
            updateHandle(e.clientX);
        }

        function stopDrag(e) {
            if (!isDragging) return;
            isDragging = false;
            if (slider.hasPointerCapture(e.pointerId)) {
                slider.releasePointerCapture(e.pointerId);
            }
        }

        applyClips();

        slider.addEventListener('pointerdown', startDrag);
        slider.addEventListener('pointermove', onDrag);
        slider.addEventListener('pointerup', stopDrag);
        slider.addEventListener('pointercancel', stopDrag);
        slider.addEventListener('lostpointercapture', function() {
            isDragging = false;
        });
    });
}

/**
 * Copy BibTeX to clipboard
 */
function initCopyBibtex() {
    // Function is called directly via onclick, but we can also attach it here
}

// Global function for copying BibTeX (called from HTML onclick)
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex-code code');
    const copyBtn = document.querySelector('.copy-btn');

    if (!bibtexCode || !copyBtn) return;

    const text = bibtexCode.textContent;

    // Use modern clipboard API if available
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showCopySuccess(copyBtn);
        }).catch(err => {
            console.error('Failed to copy:', err);
            fallbackCopy(text, copyBtn);
        });
    } else {
        fallbackCopy(text, copyBtn);
    }
}

// Fallback copy method for older browsers
function fallbackCopy(text, copyBtn) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();

    try {
        document.execCommand('copy');
        showCopySuccess(copyBtn);
    } catch (err) {
        console.error('Fallback copy failed:', err);
    }

    document.body.removeChild(textarea);
}

// Show copy success feedback
function showCopySuccess(btn) {
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    btn.classList.add('copied');

    setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.classList.remove('copied');
    }, 2000);
}

/**
 * Lazy loading for images (optional enhancement)
 */
function initLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                    }
                    observer.unobserve(img);
                }
            });
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }
}

/**
 * Animate elements on scroll (optional enhancement)
 */
function initScrollAnimations() {
    if ('IntersectionObserver' in window) {
        const animateObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll('.method-card, .highlight-item, .stat-item').forEach(el => {
            el.classList.add('animate-on-scroll');
            animateObserver.observe(el);
        });
    }
}
