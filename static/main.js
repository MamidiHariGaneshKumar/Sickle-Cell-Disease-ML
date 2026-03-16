function predictImage() {
    let input = document.getElementById("imageInput");
    if (!input.files.length) {
        alert("Please select an image!");
        return;
    }

    let formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `Prediction: ${data.prediction}, Confidence: ${data.confidence}`;
    })
    .catch(error => console.error("Error:", error));
}
