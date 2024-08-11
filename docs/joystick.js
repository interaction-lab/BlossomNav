document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('joystick-canvas');
    const ctx = canvas.getContext('2d');
    const radiusBig = Math.min(canvas.width, canvas.height) / 3; // Adjust to scale with canvas size
    const radiusSmall = radiusBig / 4;
    let centerX = canvas.width / 2;
    let centerY = canvas.height / 2;
    const maxDistance = radiusBig - radiusSmall;

    let isDragging = false;
    let currentX = centerX;
    let currentY = centerY;

    drawJoystick(centerX, centerY);

    function drawJoystick(smallCircleX, smallCircleY) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#555'; // Lighter grey for joystick background
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.arc(centerX, centerY, radiusBig, 0, Math.PI * 2, true);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(smallCircleX, smallCircleY, radiusSmall, 0, Math.PI * 2, true);
        ctx.fillStyle = 'white';
        ctx.fill();
    }

    function onMouseDown(event) {
        const { x, y } = getMousePosition(event);
        if (isInsideSmallCircle(x, y)) {
            isDragging = true;
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.setPointerCapture(event.pointerId);
        }
    }

    function onMouseMove(event) {
        if (isDragging) {
            const { x, y } = getMousePosition(event);
            const dx = x - centerX;
            const dy = y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= maxDistance) {
                currentX = x;
                currentY = y;
            } else {
                const angle = Math.atan2(dy, dx);
                currentX = centerX + maxDistance * Math.cos(angle);
                currentY = centerY + maxDistance * Math.sin(angle);
            }

            drawJoystick(currentX, currentY);
            moveRobot(currentX, currentY);
        }
    }

    function onMouseUp(event) {
        if (isDragging) {
            isDragging = false;
            drawJoystick(centerX, centerY);
            moveRobot(centerX, centerY);
            canvas.removeEventListener('mousemove', onMouseMove);
            canvas.releasePointerCapture(event.pointerId);
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

        const event = new CustomEvent('joystickMove', {
            detail: {
                x: xRelative,
                y: yRelative,
            },
        });
        
        window.dispatchEvent(event);
    }

    canvas.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp);
    window.addEventListener('resize', function () {
        const rect = canvas.getBoundingClientRect();
        centerX = rect.width / 2;
        centerY = rect.height / 2;
        drawJoystick(centerX, centerY);
    });
});
