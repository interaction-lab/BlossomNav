const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const pageUrl = 'http://192.168.1.14/picture/1/frame/'; // Ensure this is the correct URL
const outputDir = '/home/anthony/InteractionsLab/BlossomNav/data/raspberry'; // Change this to your desired directory

let captureInterval;

// Function to capture a screenshot of the entire page
async function capturePageScreenshot() {
    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        await page.goto(pageUrl, { waitUntil: 'networkidle2' });

        // Wait for the image to load
        await page.waitForSelector('img');

        // Get the current timestamp to use in the filename
        const timestamp = Date.now();

        // Define the output file path
        const outputFilePath = path.join(outputDir, `image_${timestamp}.png`);

        // Capture the screenshot of the entire page
        await page.screenshot({ path: outputFilePath, fullPage: true });

        console.log('Screenshot captured successfully:', outputFilePath);

        await browser.close();
    } catch (error) {
        console.error('Error capturing the screenshot:', error);
    }
}

// Function to start capturing screenshots every 0.2 seconds
function startCapturing() {
    if (captureInterval) {
        console.log('Capturing is already in progress.');
        return;
    }
    captureInterval = setInterval(capturePageScreenshot, 500);
    console.log('Started capturing screenshots every 0.5 seconds.');
}

// Function to stop capturing screenshots
function stopCapturing() {
    if (!captureInterval) {
        console.log('Capturing is not in progress.');
        return;
    }
    clearInterval(captureInterval);
    captureInterval = null;
    console.log('Stopped capturing screenshots.');
}

// Expose start and stop functions to be called externally
module.exports = {
    startCapturing,
    stopCapturing
};

// Example usage (uncomment these lines to test directly in this script):
startCapturing();
setTimeout(stopCapturing, 20000); // Stops after 10 seconds