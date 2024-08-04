document.addEventListener('DOMContentLoaded', function () {
    const demoCanvas = document.getElementById('robot-demo');
    const demoCtx = demoCanvas.getContext('2d');

    // Set canvas dimensions
    demoCanvas.width = 300; // Width of the environment
    demoCanvas.height = 300; // Height of the environment

    const robotSize = 20; // Size of the robot triangle
    const centerX = demoCanvas.width / 2;
    const centerY = demoCanvas.height / 2;
    let robotAngle = 0; // Initial angle
    let robotX = centerX;
    let robotY = centerY;

    // Draw the robot and obstacles initially
    drawEnvironment();
    drawRobot(robotX, robotY, robotAngle);

    function drawEnvironment() {
        demoCtx.fillStyle = '#fff';
        demoCtx.fillRect(0, 0, demoCanvas.width, demoCanvas.height);

        // Draw obstacles
        demoCtx.fillStyle = '#000';

        // Rectangle obstacle 1
        demoCtx.fillRect(50, 75, 50, 25);

        // Rectangle obstacle 2
        demoCtx.fillRect(200, 150, 50, 25);

        // L-shaped desk 1
        demoCtx.fillRect(100, 200, 10, 50); // Vertical part
        demoCtx.fillRect(100, 240, 30, 10); // Horizontal part

        // L-shaped desk 2
        demoCtx.fillRect(250, 50, 10, 50); // Vertical part
        demoCtx.fillRect(230, 50, 30, 10); // Horizontal part
    }

    function drawRobot(x, y, angle) {
        demoCtx.clearRect(0, 0, demoCanvas.width, demoCanvas.height);
        drawEnvironment();

        demoCtx.save();
        demoCtx.translate(x, y);
        demoCtx.rotate(angle);

        // Draw robot as a triangle with a red tip
        demoCtx.beginPath();
        demoCtx.moveTo(0, -robotSize); // Tip of the triangle
        demoCtx.lineTo(robotSize / 2, robotSize);
        demoCtx.lineTo(-robotSize / 2, robotSize);
        demoCtx.closePath();

        // Fill the triangle with a grey body
        demoCtx.fillStyle = 'grey';
        demoCtx.fill();

        // Red tip
        demoCtx.beginPath();
        demoCtx.moveTo(0, -robotSize);
        demoCtx.lineTo(-robotSize / 6, -robotSize / 3);
        demoCtx.lineTo(robotSize / 6, -robotSize / 3);
        demoCtx.closePath();

        demoCtx.fillStyle = 'red';
        demoCtx.fill();

        demoCtx.restore();
    }

    function updateRobotPosition(dx, dy) {
        const speed = 2; // Speed of movement
        const turnSpeed = 0.05; // Speed of rotation

        // Update robot position and angle based on joystick input
        robotAngle += dx * turnSpeed; // Turn left/right
        robotX += Math.sin(robotAngle) * dy * speed; // Move forward/backward based on y direction
        robotY -= Math.cos(robotAngle) * dy * speed;

        // Ensure the robot stays within bounds
        robotX = Math.max(robotSize, Math.min(robotX, demoCanvas.width - robotSize));
        robotY = Math.max(robotSize, Math.min(robotY, demoCanvas.height - robotSize));
    }

    // Listen for joystick movements
    window.addEventListener('joystickMove', function (event) {
        const { x, y } = event.detail;
        updateRobotPosition(x, y);
        drawRobot(robotX, robotY, robotAngle);
    });
});
