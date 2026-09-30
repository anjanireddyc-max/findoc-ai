const fileInput = document.getElementById("documentInput");
const fileName = document.getElementById("fileName");

const startCamera = document.getElementById("startCamera");
const closeCamera = document.getElementById("closeCamera");

const cameraArea = document.getElementById("cameraArea");
const cameraVideo = document.getElementById("cameraVideo");

const captureButton = document.getElementById("captureButton");

const previewArea = document.getElementById("previewArea");
const capturedImage = document.getElementById("capturedImage");

const retakeButton = document.getElementById("retakeButton");

const canvas = document.getElementById("cameraCanvas");

let cameraStream = null;


// =========================================================
// FILE UPLOAD
// =========================================================

fileInput.addEventListener("change", function () {

    if (this.files.length > 0) {

        fileName.textContent =
            this.files[0].name;

    } else {

        fileName.textContent =
            "No file selected";
    }

});


// =========================================================
// START CAMERA
// =========================================================

startCamera.addEventListener(
    "click",
    async function () {

        try {

            cameraStream =
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "environment"
                    },
                    audio: false
                });


            cameraVideo.srcObject =
                cameraStream;


            cameraArea.style.display =
                "block";

            previewArea.style.display =
                "none";

            startCamera.style.display =
                "none";

        } catch (error) {

            console.error(error);

            alert(
                "Camera access failed. Please allow camera permission and try again."
            );

        }

    }
);


// =========================================================
// CAPTURE
// =========================================================

captureButton.addEventListener(
    "click",
    function () {

        if (!cameraStream) {

            alert("Camera is not active.");

            return;
        }


        const width =
            cameraVideo.videoWidth;

        const height =
            cameraVideo.videoHeight;


        canvas.width = width;
        canvas.height = height;


        const context =
            canvas.getContext("2d");


        context.drawImage(
            cameraVideo,
            0,
            0,
            width,
            height
        );


        canvas.toBlob(
            function (blob) {

                const capturedFile =
                    new File(
                        [blob],
                        "camera_capture.jpg",
                        {
                            type: "image/jpeg"
                        }
                    );


                const dataTransfer =
                    new DataTransfer();


                dataTransfer.items.add(
                    capturedFile
                );


                fileInput.files =
                    dataTransfer.files;


                fileName.textContent =
                    "Camera capture ready";


                capturedImage.src =
                    URL.createObjectURL(blob);


                previewArea.style.display =
                    "block";


                cameraArea.style.display =
                    "none";


                stopCamera();

            },
            "image/jpeg",
            0.95
        );

    }
);


// =========================================================
// CLOSE CAMERA
// =========================================================

closeCamera.addEventListener(
    "click",
    function () {

        stopCamera();

        cameraArea.style.display =
            "none";

        startCamera.style.display =
            "inline-flex";

    }
);


// =========================================================
// RETAKE
// =========================================================

retakeButton.addEventListener(
    "click",
    async function () {

        previewArea.style.display =
            "none";

        startCamera.style.display =
            "none";

        try {

            cameraStream =
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "environment"
                    },
                    audio: false
                });


            cameraVideo.srcObject =
                cameraStream;


            cameraArea.style.display =
                "block";

        } catch (error) {

            alert(
                "Unable to reopen camera."
            );

        }

    }
);


// =========================================================
// STOP CAMERA
// =========================================================

function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;
    }

    cameraVideo.srcObject = null;
}