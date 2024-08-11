document.addEventListener('DOMContentLoaded', function () {
    const demoCanvas = document.getElementById('robot-demo');
    const demoCtx = demoCanvas.getContext('2d');

    const robotSize = Math.min(demoCanvas.width, demoCanvas.height) / 15; // Adjust to scale with canvas size
    let centerX = demoCanvas.width / 2;
    let centerY = demoCanvas.height / 2;
    let robotAngle = 0;
    let robotX = centerX;
    let robotY = centerY;

    drawEnvironment();
    drawRobot(robotX, robotY, robotAngle);

    function drawEnvironment() {
        demoCtx.fillStyle = '#fff';
        demoCtx.fillRect(0, 0, demoCanvas.width, demoCanvas.height);

        demoCtx.fillStyle = '#000';
        demoCtx.fillRect(50, 75, demoCanvas.width * 0.2, demoCanvas.height * 0.1);
        demoCtx.fillRect(demoCanvas.width * 0.7, demoCanvas.height * 0.5, demoCanvas.width * 0.2, demoCanvas.height * 0.1);
        demoCtx.fillRect(100, 200, demoCanvas.width * 0.03, demoCanvas.height * 0.17);
        demoCtx.fillRect(100, 240, demoCanvas.width * 0.1, demoCanvas.height * 0.03);
        demoCtx.fillRect(250, 50, demoCanvas.width * 0.03, demoCanvas.height * 0.17);
        demoCtx.fillRect(230, 50, demoCanvas.width * 0.1, demoCanvas.height * 0.03);
    }

    function drawRobot(x, y, angle) {
        demoCtx.clearRect(0, 0, demoCanvas.width, demoCanvas.height);
        drawEnvironment();

        demoCtx.save();
        demoCtx.translate(x, y);
        demoCtx.rotate(angle);

        const height = robotSize * Math.sqrt(3) / 2;
        demoCtx.beginPath();
        demoCtx.moveTo(0, -height);
        demoCtx.lineTo(robotSize / 2, height / 2);
        demoCtx.lineTo(-robotSize / 2, height / 2);
        demoCtx.closePath();

        demoCtx.fillStyle = 'grey';
        demoCtx.fill();

        demoCtx.restore();
    }

    function updateRobotPosition(dx, dy) {
        const speed = 2;
        const turnSpeed = 0.05;

        let newX = robotX + Math.sin(robotAngle) * dy * speed;
        let newY = robotY - Math.cos(robotAngle) * dy * speed;

        if (!checkCollision(newX, newY)) {
            robotX = newX;
            robotY = newY;
        }

        robotAngle += dx * turnSpeed;

        robotX = Math.max(robotSize, Math.min(robotX, demoCanvas.width - robotSize));
        robotY = Math.max(robotSize, Math.min(robotY, demoCanvas.height - robotSize));
    }

    function checkCollision(x, y) {
        const obstacles = [
            { x: 50, y: 75, width: demoCanvas.width * 0.2, height: demoCanvas.height * 0.1 },
            { x: demoCanvas.width * 0.7, y: demoCanvas.height * 0.5, width: demoCanvas.width * 0.2, height: demoCanvas.height * 0.1 },
            { x: 100, y: 200, width: demoCanvas.width * 0.03, height: demoCanvas.height * 0.17 },
            { x: 100, y: 240, width: demoCanvas.width * 0.1, height: demoCanvas.height * 0.03 },
            { x: 250, y: 50, width: demoCanvas.width * 0.03, height: demoCanvas.height * 0.17 },
            { x: 230, y: 50, width: demoCanvas.width * 0.1, height: demoCanvas.height * 0.03 }
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

    window.addEventListener('joystickMove', function (event) {
        const { x, y } = event.detail;
        updateRobotPosition(x, y);
        drawRobot(robotX, robotY, robotAngle);
    });
});
