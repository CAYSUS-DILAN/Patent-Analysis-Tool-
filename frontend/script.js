const socket = io.connect("http://127.0.0.1:5000");

document.getElementById("uploadBtn").addEventListener("click", function () {
    let fileInput = document.getElementById("fileInput");
    let keywordInput = document.getElementById("keywordInput");

    if (!fileInput.files.length || !keywordInput.value.trim()) {
        alert("Please upload a file and enter keywords.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("keywords", keywordInput.value);

    fetch("/upload", {
        method: "POST",
        body: formData
    }).then(response => response.json())
      .then(data => {
          if (data.error) {
              alert(data.error);
          } else {
              alert(data.message);
          }
      });
});

socket.on("update_progress", function (data) {
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");

    progressBar.style.width = data.progress + "%";
    progressBar.innerText = data.progress + "%";
    progressText.textContent = ${data.progress}% (${data.processed_urls}/${data.total_urls} URLs processed);
});

socket.on("processing_ended", function () {
    const keywords = document.getElementById("keywordInput").value.split(",");
    const viewResultsBtn = document.getElementById("viewResultsBtn");

    // Store keywords in localStorage
    localStorage.setItem("keywords", keywords.join(","));

    // Show the "View Results" button
    viewResultsBtn.style.display = "block";
});

socket.on("show_download_button", function () {
    const downloadBtn = document.getElementById("downloadBtn");
    downloadBtn.style.display = "block";
});

let isPaused = false;
document.getElementById("pauseBtn").addEventListener("click", function () {
    isPaused = !isPaused;
    if (isPaused) {
        socket.emit("pause");
        this.textContent = "Resume";
    } else {
        socket.emit("resume");
        this.textContent = "Pause";
    }
});

document.getElementById("viewResultsBtn").addEventListener("click", function () {
    window.location.href = 'result.html';
});

document.getElementById("downloadBtn").addEventListener("click", function () {
    const keywords = document.getElementById("keywordInput").value.split(",");
    fetch("/download_results", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ keywords: keywords })
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error("Network response was not ok.");
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = "results.zip";
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        console.error("There was an error downloading the results:", error);
    });
});

document.getElementById("logoutBtn").addEventListener("click", function () {
    fetch("/logout", {
        method: "POST"
    }).then(response => {
        if (response.ok) {
            window.location.href = 'login.html';
        }
    });
});