// The URL of the streaming image
const imageUrl = 'http://192.168.1.14:8081/';

// Create an Image object
const image = new Image();

// Set the crossOrigin attribute if needed (e.g., if the image is hosted on a different domain)
image.crossOrigin = 'anonymous';

// Set the image source to the streaming URL
image.src = imageUrl;

// Once the image loads, draw it onto a canvas and trigger the download
image.onload = function() {
    // Create a canvas element
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;

    // Get the canvas context
    const ctx = canvas.getContext('2d');

    // Draw the image onto the canvas
    ctx.drawImage(image, 0, 0);

    // Convert the canvas content to a data URL
    const dataUrl = canvas.toDataURL('image/jpeg'); // or 'image/png'

    // Create a temporary anchor element
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = 'downloaded_image.jpg'; // Specify the file name and extension

    // Append the anchor to the body
    document.body.appendChild(link);

    // Simulate a click on the anchor to trigger the download
    link.click();

    // Remove the anchor from the document
    document.body.removeChild(link);
};

image.onerror = function() {
    console.error('Error loading the image.');
};
