document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("file-input");
  const imagePreview = document.getElementById("image-preview");
  const cropSend = document.getElementById("crop-send");
  const summaryBox = document.getElementById("summary");
  const retrievedBox = document.getElementById("retrieved");
  const regenerateBtn = document.getElementById("regenerate");
  const manualSearch = document.getElementById("analyze-name");
  const manualInput = document.getElementById("manual-name");

  let cropper;
  let lastNames = []; // track all searched medicine names

  // ========== Image Upload ==========
  fileInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (evt) {
      imagePreview.src = evt.target.result;
      imagePreview.style.display = "block";
      cropSend.style.display = "inline-block";
      if (cropper) cropper.destroy();
      cropper = new Cropper(imagePreview, { aspectRatio: NaN });
    };
    reader.readAsDataURL(file);
  });

  // ========== Crop & Analyze ==========
  cropSend.addEventListener("click", function () {
    if (!cropper) return;
    const canvas = cropper.getCroppedCanvas();
    const base64 = canvas.toDataURL("image/png");
    summaryBox.textContent = "🔍 Analyzing prescription...";
    retrievedBox.innerHTML = "";
    fetch("/analyze-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    })
      .then((r) => r.json())
      .then((data) => {
        lastNames = (data.summaries || []).map(s => s.drug_name);
        renderResponse(data);
      })
      .catch((err) => {
        summaryBox.textContent = "❌ Error analyzing image: " + err;
      });
  });

  // ========== Manual Name Search ==========
  manualSearch.addEventListener("click", function () {
    const name = manualInput.value.trim();
    if (!name) {
      alert("Please enter a medicine name.");
      return;
    }
    summaryBox.textContent = "🔍 Fetching details...";
    retrievedBox.innerHTML = "";
    fetch("/analyze-name", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    })
      .then((r) => r.json())
      .then((data) => {
        lastNames = (data.summaries || []).map(s => s.drug_name);
        renderResponse(data);
      })
      .catch((err) => {
        summaryBox.textContent = "❌ Error: " + err;
      });
  });

  // ========== Regenerate Summary ==========
  regenerateBtn.addEventListener("click", async function () {
    if (!lastNames.length) return;

    summaryBox.textContent = "🔄 Regenerating summaries...";
    const newSummaries = [];
    const newRetrieved = [];

    for (const name of lastNames) {
      try {
        const res = await fetch("/regenerate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        const data = await res.json();
        if (data.ok) {
          newSummaries.push(...data.summaries);
          newRetrieved.push(...data.retrieved);
        }
      } catch (err) {
        console.error("Error regenerating", name, err);
      }
    }

    renderResponse({ summaries: newSummaries, retrieved: newRetrieved, ok: true });
  });

  // ========== Renderer ==========
  function renderResponse(data) {
    if (!data.ok) {
      summaryBox.textContent = data.error || data.message || "❌ Unknown error.";
      regenerateBtn.style.display = "none";
      return;
    }

    // ⚠️ Prominent warning at the top (once)
    let summaryHTML = "";
    if ((data.summaries || []).length > 0) {
      summaryHTML += `<div style="font-size:1.2em; font-weight:bold; color:red; margin-bottom:10px;">
        ⚠️⚠️ WARNING: The results may include closest matching medicines by name. Please verify carefully before use! ⚠️⚠️
      </div>`;
    }

    // 1️⃣ Show summaries
    (data.summaries || []).forEach((s) => {
      const formattedSummary = s.summary.replace(/\n/g, "<br>");
      summaryHTML += `🧾 <b>${s.drug_name}</b>: ${formattedSummary}<br><br>`;
    });

    summaryBox.innerHTML = summaryHTML || "No summaries generated.";
    regenerateBtn.style.display = "inline-block";

    // 2️⃣ Show medicine details + embedded alternatives
    retrievedBox.innerHTML = "";
    (data.retrieved || []).forEach((item) => {
      let html = `<div class="medicine-detail">
        <h4>💊 ${item["Drug Name"] || "Unknown"}</h4>
        <p><b>Company:</b> ${item["Company Name"] || "N/A"}</p>
        <p><b>Ingredient:</b> ${item["Active Ingredient"] || "N/A"}</p>
        <p><b>Use:</b> ${item["Indication"] || "N/A"}</p>
        <p><b>Dosage:</b> ${item["Dosage and Administration"] || "N/A"}</p>
        <p><b>Side Effects:</b> ${item["Side Effects"] || "N/A"}</p>
        <p><b>Pregnancy:</b> ${item["Use in pregnancy"] || "N/A"}</p>`;

      const alts = item["Alternative Medicines"] || [];
      if (alts.length > 0) {
        html += `<p><b>🔄 Alternatives:</b> ${alts.map((a) => a["Drug Name"]).join(", ")}</p>`;
      } else {
        html += `<p><b>🔄 Alternatives:</b> 🙏 No alternates available</p>`;
      }

      html += "</div><hr>";
      retrievedBox.innerHTML += html;
    });
  }

  // ========== Slideshow ==========
  let slideIndex = 0;
  const slides = document.querySelectorAll(".slides");
  const dots = document.querySelectorAll(".dot");

  function showSlides(n) {
    if (slides.length === 0) return;
    if (n >= slides.length) slideIndex = 0;
    if (n < 0) slideIndex = slides.length - 1;

    slides.forEach((slide) => (slide.style.display = "none"));
    dots.forEach((dot) => dot.classList.remove("active"));

    slides[slideIndex].style.display = "block";
    dots[slideIndex].classList.add("active");
  }

  function nextSlide() {
    slideIndex++;
    showSlides(slideIndex);
  }

  function currentSlide(n) {
    slideIndex = n - 1; // dots are 1-indexed
    showSlides(slideIndex);
  }

  setInterval(nextSlide, 5000); // auto change every 5s
  showSlides(slideIndex);
  window.currentSlide = currentSlide; // make accessible for HTML onclick
});
