const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const confidenceBox = document.getElementById('confidence-box');
const confidenceList = document.getElementById('confidence-list');

let currentQuestionCode = null;
let isInitialDemographics = true;

// Function to add a message to the chat box
function addMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    const messageP = document.createElement('p');
    messageP.textContent = message;
    messageDiv.appendChild(messageP);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

// Function to handle errors
function handleError(error) {
    console.error('API Error:', error);
    addMessage('Sorry, something went wrong. Please try again later.', 'bot');
}

// Function to update confidence display
function updateConfidence(confidenceData) {
    if (confidenceData && confidenceData.confidence_levels) {
        confidenceList.innerHTML = ''; // Clear previous list
        confidenceData.confidence_levels.forEach(item => {
            const li = document.createElement('li');
            li.textContent = `${item.disease}: ${(item.confidence * 100).toFixed(1)}%`;
            confidenceList.appendChild(li);
        });
        confidenceBox.style.display = 'block';
    } else {
        confidenceBox.style.display = 'none';
    }
}

// Function to get diagnosis confidence
async function getConfidence() {
    try {
        const response = await fetch('/confidence');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === 'success') {
            updateConfidence(data);
        } else {
            console.warn('Could not get confidence:', data.error || 'Unknown reason');
            confidenceBox.style.display = 'none';
        }
    } catch (error) {
        handleError(error);
    }
}

// Function to get the next question
async function getNextQuestion() {
    try {
        const response = await fetch('/question');
        if (!response.ok) {
            const errorData = await response.json();
            // Handle specific cases like max turns reached
            if (errorData.status === 'complete') {
                addMessage(errorData.message, 'bot');
                getConfidence(); // Show final confidence
                userInput.disabled = true;
                sendButton.disabled = true;
                userInput.placeholder = "Conversation finished.";
            } else {
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error}`);
            }
            return; 
        }
        const data = await response.json();
        if (data.code && data.question_en) {
            addMessage(data.question_en, 'bot');
            currentQuestionCode = data.code;
            userInput.placeholder = "Your answer...";
        } else {
            // Handle potential end of conversation or other issues
            addMessage("Thank you. The diagnosis process is complete.", "bot");
            getConfidence(); // Show final confidence
            userInput.disabled = true;
            sendButton.disabled = true;
            userInput.placeholder = "Conversation finished.";
        }
    } catch (error) {
        handleError(error);
        userInput.placeholder = "Error occurred. Cannot proceed.";
        userInput.disabled = true;
        sendButton.disabled = true;
    }
}

// Function to start the session
async function startSession(age, sex) {
    try {
        const response = await fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // Include age and sex if needed by the backend logic, though current /start doesn't use them
            // body: JSON.stringify({ age: parseInt(age), sex: sex })
            body: JSON.stringify({}) // Send empty body as /start doesn't expect data
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.message === "Session started successfully") {
            isInitialDemographics = false;
            userInput.placeholder = "Loading first question...";
            await getNextQuestion(); // Get the first question after starting
        } else {
            addMessage("Failed to start the session. Please refresh.", 'bot');
        }
    } catch (error) {
        handleError(error);
    }
}

// Function to handle sending messages
async function sendMessage() {
    const messageText = userInput.value.trim();
    if (!messageText) return;

    addMessage(messageText, 'user');
    userInput.value = ''; // Clear input field

    if (isInitialDemographics) {
        // Parse age and sex
        const parts = messageText.match(/(\d+)\s+(male|female)/i);
        if (parts && parts.length === 3) {
            const age = parts[1];
            const sex = parts[2].toLowerCase();
            userInput.placeholder = "Starting session...";
            userInput.disabled = true;
            sendButton.disabled = true;
            await startSession(age, sex);
            userInput.disabled = false;
            sendButton.disabled = false;
        } else {
            addMessage("Please enter your age and sex in the format 'age sex' (e.g., '30 male').", 'bot');
        }
        return;
    }

    if (currentQuestionCode) {
        userInput.placeholder = "Processing answer...";
        userInput.disabled = true;
        sendButton.disabled = true;
        try {
            const response = await fetch('/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code: currentQuestionCode, answer: messageText }),
            });
            if (!response.ok) {
                 const errorData = await response.json();
                 throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error}`);
            }
            const data = await response.json();
            if (data.message === "Answer recorded successfully") {
                await getConfidence(); // Update confidence after answer
                await getNextQuestion(); // Ask next question
            } else {
                 addMessage(data.error || "Failed to record answer.", 'bot');
            }
        } catch (error) {
            handleError(error);
        }
        userInput.disabled = false;
        sendButton.disabled = false;
    } else {
        // Should not happen if flow is correct, but handle it
        addMessage("Waiting for the next question...", 'bot');
        getNextQuestion(); // Try getting a question again
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

// Initial setup - No need to call /start automatically, prompt user first
// addMessage("Welcome! Please provide your age and sex to start.", "bot"); 