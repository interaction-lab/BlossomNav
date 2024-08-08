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

        // Draw robot as an equilateral triangle
        const height = robotSize * Math.sqrt(3) / 2; // Height for equilateral triangle
        demoCtx.beginPath();
        demoCtx.moveTo(0, -height); // Top vertex
        demoCtx.lineTo(robotSize / 2, height / 2); // Bottom right
        demoCtx.lineTo(-robotSize / 2, height / 2); // Bottom left
        demoCtx.closePath();

        demoCtx.fillStyle = 'grey';
        demoCtx.fill();

        demoCtx.restore();
    }

    function updateRobotPosition(dx, dy) {
        const speed = 2; // Speed of movement
        const turnSpeed = 0.05; // Speed of rotation

        // Calculate new position
        let newX = robotX + Math.sin(robotAngle) * dy * speed;
        let newY = robotY - Math.cos(robotAngle) * dy * speed;

        // Check for collisions with obstacles
        if (!checkCollision(newX, newY)) {
            robotX = newX;
            robotY = newY;
        }

        // Update angle
        robotAngle += dx * turnSpeed; // Turn left/right

        // Ensure the robot stays within bounds
        robotX = Math.max(robotSize, Math.min(robotX, demoCanvas.width - robotSize));
        robotY = Math.max(robotSize, Math.min(robotY, demoCanvas.height - robotSize));
    }

    function checkCollision(x, y) {
        const obstacles = [
            { x: 50, y: 75, width: 50, height: 25 },
            { x: 200, y: 150, width: 50, height: 25 },
            { x: 100, y: 200, width: 10, height: 50 },
            { x: 100, y: 240, width: 30, height: 10 },
            { x: 250, y: 50, width: 10, height: 50 },
            { x: 230, y: 50, width: 30, height: 10 }
        ];

        return obstacles.some(obstacle => {
            return (
                x + robotSize / 2 > obstacle.x &&
                x - robotSize / 2 < obstacle.x + obstacle.width &&
                y + robotSize / 2 > obstacle.y &&
                y - robotSize / 2 < obstacle.y + obstacle.height
            );
        });
    }

    // Listen for joystick movements
    window.addEventListener('joystickMove', function (event) {
        const { x, y } = event.detail;
        updateRobotPosition(x, y);
        drawRobot(robotX, robotY, robotAngle);
    });
});
