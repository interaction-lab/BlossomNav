document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('joystick-canvas');
    const ctx = canvas.getContext('2d');
    const radiusBig = 145; // Radius of the big circle
    const radiusSmall = 20; // Radius of the small circle
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const maxDistance = radiusBig - radiusSmall; // Max distance the small circle can move from the center

    let isDragging = false;
    let currentX = centerX;
    let currentY = centerY;

    // Draw the initial joystick
    drawJoystick(centerX, centerY);

    function drawJoystick(smallCircleX, smallCircleY) {
        // Set canvas background color to match the website background
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#333'; // Dark grey
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw big circle (outer boundary)
        ctx.beginPath();
        ctx.arc(centerX, centerY, radiusBig, 0, Math.PI * 2, true);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw small circle (joystick handle)
        ctx.beginPath();
        ctx.arc(smallCircleX, smallCircleY, radiusSmall, 0, Math.PI * 2, true);
        ctx.fillStyle = 'white';
        ctx.fill();
    }

    function onMouseDown(event) {
        const { x, y } = getMousePosition(event);
        if (isInsideSmallCircle(x, y)) {
            isDragging = true;
            canvas.addEventListener('mousemove', onMouseMove); // Ensure mousemove listener is added
            canvas.setPointerCapture(event.pointerId); // Capture the pointer to ensure events are captured
        }
    }

    function onMouseMove(event) {
        if (isDragging) {
            const { x, y } = getMousePosition(event);
            const dx = x - centerX;
            const dy = y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= maxDistance) {
                // Move freely within the circle
                currentX = x;
                currentY = y;
            } else {
                // Constrain to the edge
                const angle = Math.atan2(dy, dx);
                currentX = centerX + maxDistance * Math.cos(angle);
                currentY = centerY + maxDistance * Math.sin(angle);
            }

            drawJoystick(currentX, currentY);
            moveRobot(currentX, currentY); // Call robot control function
        }
    }

    function onMouseUp(event) {
        if (isDragging) {
            isDragging = false;
            drawJoystick(centerX, centerY); // Reset to center
            moveRobot(centerX, centerY); // Reset robot position
            canvas.removeEventListener('mousemove', onMouseMove); // Remove listener when not dragging
            canvas.releasePointerCapture(event.pointerId); // Release pointer capture
        }
    }

    function getMousePosition(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    function isInsideSmallCircle(x, y) {
        const dx = x - currentX;
        const dy = y - currentY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance <= radiusSmall;
    }

    function moveRobot(x, y) {
        const xRelative = (x - centerX) / maxDistance;
        const yRelative = (centerY - y) / maxDistance;

        // Trigger robot movement
        const event = new CustomEvent('joystickMove', {
            detail: {
                x: xRelative,
                y: yRelative,
            },
        });

        window.dispatchEvent(event);
    }

    // Add event listeners
    canvas.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp); // Ensure mouseup is on document for reliability
    window.addEventListener('resize', function() {
        // Re-calculate center position on window resize
        const rect = canvas.getBoundingClientRect();
        centerX = rect.width / 2;
        centerY = rect.height / 2;
        drawJoystick(centerX, centerY);
    });
});
