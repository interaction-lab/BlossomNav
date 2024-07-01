const axios = require('axios');
const cheerio = require('cheerio');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

// URL of the streaming page
const pageUrl = '';

// Function to fetch HTML content from a URL
async function fetchPageHtml(url) {
    try {
        const response = await axios.get(url);
        return response.data;
    } catch (error) {
        console.error('Error fetching the HTML content:', error);
        return null;
    }
}

// Function to extract image URL from HTML
function extractImageUrl(html) {
    const $ = cheerio.load(html);
    const imgElement = $('img');
    return imgElement.attr('src');
}

// Function to download the image to a specified directory
async function downloadImage(imageUrl, outputDir) {
    try {
        // Ensure imageUrl is a valid absolute URL
        const absoluteUrl = new URL(imageUrl, pageUrl).toString();

        const response = await fetch(absoluteUrl);
        const buffer = await response.buffer();

        // Extracting filename from URL
        const filename = path.basename(new URL(absoluteUrl).pathname);

        // Construct the output file path
        const outputFilePath = path.join(outputDir, filename);

        // Write the buffer to the specified directory
        fs.writeFileSync(outputFilePath, buffer);
        console.log('Image downloaded successfully to:', outputFilePath);
    } catch (error) {
        console.error('Error downloading the image:', error);
    }
}

// Main function to orchestrate the process
async function main() {
    const html = await fetchPageHtml(pageUrl);
    if (html) {
        const imageUrl = extractImageUrl(html);
        if (imageUrl) {
            console.log('Image URL found:', imageUrl);
            const outputDir = '/home/anthony/InteractionsLab/BlossomNav/data/scrapped'; // Change this to your desired directory
            await downloadImage(imageUrl, outputDir);
        } else {
            console.error('No image URL found in the HTML.');
        }
    } else {
        console.error('Failed to fetch HTML content.');
    }
}

// Execute the main function
main();
