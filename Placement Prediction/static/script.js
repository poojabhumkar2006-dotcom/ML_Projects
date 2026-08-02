/* ==========================================
   AI Placement Predictor
   script.js
========================================== */

// ==========================
// AOS Animation
// ==========================

AOS.init({
    duration: 1000,
    once: true
});

// ==========================
// Animated Counters
// ==========================

document.querySelectorAll(".counter").forEach(counter => {

    const target = Number(counter.innerText);

    let count = 0;

    const speed = Math.max(1, Math.ceil(target / 60));

    function updateCounter() {

        if (count < target) {

            count += speed;

            if (count > target) count = target;

            counter.innerText = count;

            requestAnimationFrame(updateCounter);

        }

    }

    updateCounter();

});

// ==========================
// Navbar Scroll Effect
// ==========================

window.addEventListener("scroll", function () {

    const navbar = document.querySelector(".navbar");

    if (window.scrollY > 50) {

        navbar.style.background = "rgba(15,23,42,0.95)";

    } else {

        navbar.style.background = "rgba(15,23,42,0.85)";

    }

});

// ==========================
// Smooth Scrolling
// ==========================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {

    anchor.addEventListener("click", function (e) {

        e.preventDefault();

        const target = document.querySelector(this.getAttribute("href"));

        if (target) {

            target.scrollIntoView({

                behavior: "smooth"

            });

        }

    });

});

// ==========================
// Search Prediction History
// ==========================

const searchBox = document.getElementById("historySearch");

if (searchBox) {

    searchBox.addEventListener("keyup", function () {

        const value = this.value.toLowerCase();

        document.querySelectorAll("#historyTable tbody tr").forEach(row => {

            row.style.display = row.innerText.toLowerCase().includes(value)
                ? ""
                : "none";

        });

    });

}

// ==========================
// Progress Bar Animation
// ==========================

document.querySelectorAll(".progress-bar").forEach(bar => {

    const width = bar.style.width;

    bar.style.width = "0%";

    setTimeout(() => {

        bar.style.width = width;

    }, 400);

});

// ==========================
// Charts
// ==========================

const pieCanvas = document.getElementById("pieChart");

if (pieCanvas) {

    new Chart(pieCanvas, {

        type: "pie",

        data: {

            labels: ["Placed", "Not Placed"],

            datasets: [{

                data: [

                    Number(document.querySelectorAll(".counter")[1]?.innerText || 0),

                    Number(document.querySelectorAll(".counter")[2]?.innerText || 0)

                ]

            }]

        },

        options: {

            responsive: true,

            plugins: {

                legend: {

                    position: "bottom"

                }

            }

        }

    });

}

const barCanvas = document.getElementById("barChart");

if (barCanvas) {

    new Chart(barCanvas, {

        type: "bar",

        data: {

            labels: ["Placed", "Not Placed"],

            datasets: [{

                label: "Students",

                data: [

                    Number(document.querySelectorAll(".counter")[1]?.innerText || 0),

                    Number(document.querySelectorAll(".counter")[2]?.innerText || 0)

                ]

            }]

        },

        options: {

            responsive: true,

            scales: {

                y: {

                    beginAtZero: true

                }

            }

        }

    });

}

const lineCanvas = document.getElementById("lineChart");

if (lineCanvas) {

    new Chart(lineCanvas, {

        type: "line",

        data: {

            labels: [

                "Jan",

                "Feb",

                "Mar",

                "Apr",

                "May",

                "Jun"

            ],

            datasets: [{

                label: "Predictions",

                data: [

                    10,

                    15,

                    20,

                    18,

                    30,

                    Number(document.querySelectorAll(".counter")[0]?.innerText || 0)

                ],

                tension: 0.4,

                fill: true

            }]

        },

        options: {

            responsive: true

        }

    });

}

console.log("AI Placement Predictor Loaded Successfully!");