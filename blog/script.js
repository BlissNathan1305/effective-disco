const menuToggle = document.getElementById("menu-toggle");
const navbar = document.getElementById("navbar").querySelector("ul");

menuToggle.addEventListener("click", () => {
  navbar.classList.toggle("show");
});

// Smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute("href"));
    target.scrollIntoView({ behavior: "smooth" });
    navbar.classList.remove("show");
  });
});
