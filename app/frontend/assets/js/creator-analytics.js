document.addEventListener('DOMContentLoaded', () => {

    // --- CONFIGURATION ---
    const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

    // --- ELEMENT SELECTORS ---
    const mainContent = document.getElementById('main-content-area');
    const loadingState = document.getElementById('loading-state');
    const loadingStatusText = document.getElementById('loading-status-text');

    // --- DATA BINDING FUNCTIONS ---
    // These functions link our data to the specific HTML elements
    const ui = {
        setBackground: (url) => { document.getElementById('hero-banner-bg').style.backgroundImage = `url('${url}')`; },
        setAvatar: (url) => { document.getElementById('creator-avatar').style.backgroundImage = `url('${url}')`; },
        setName: (name) => { document.getElementById('creator-name').textContent = name; },
        setHandle: (handle) => { document.getElementById('creator-handle').textContent = handle; },
        setStatFollowers: (text) => { document.getElementById('stats-followers').textContent = text; },
        setStatAvgViews: (text) => { document.getElementById('stats-avg-views').textContent = text; },
        setStatEngagement: (text) => { document.getElementById('stats-engagement').textContent = text; },
        setStatVideos: (text) => { document.getElementById('stats-videos').textContent = text; },
        setQuickStatViews: (text) => { document.getElementById('quick-stats-total-views').textContent = text; },
        setQuickStatSubs: (text) => { document.getElementById('quick-stats-subscribers').textContent = text; },
        setQuickStatLikes: (text) => { document.getElementById('quick-stats-avg-likes').textContent = text; },
        setQuickStatComments: (text) => { document.getElementById('quick-stats-avg-comments').textContent = text; },
    };

    // --- UTILITY FUNCTIONS ---
    const formatNumber = (num) => {
        if (num === null || num === undefined) return 'N/A';
        if (num >= 1_000_000_000) return (num / 1_000_000_000).toFixed(1) + 'B';
        if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';
        if (num >= 1_000) return (num / 1_000).toFixed(0) + 'K';
        return num.toString();
    };

    const getChannelIdFromUrl = () => {
        const params = new URLSearchParams(window.location.search);
        return params.get('channel_id');
    };

    // --- CORE API LOGIC ---

    const startCollectionAndPoll = async (channelId) => {
        try {
            // Step 1: Trigger the collection job
            loadingStatusText.textContent = 'Starting data collection...';
            const collectApiUrl = `${API_BASE_URL}/collect/youtube-channel`;
            const collectResponse = await fetch(collectApiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ channel_id: channelId, include_detailed_stats: true, max_videos: 50 }), // Collect recent 50 videos
            });

            if (collectResponse.status !== 202) {
                const errorData = await collectResponse.json();
                throw new Error(errorData.detail || 'Failed to start analysis task.');
            }

            const taskData = await collectResponse.json();
            const taskId = taskData.task_id;

            // Step 2: Poll for the result
            loadingStatusText.textContent = 'Collecting videos (this may take a moment)...';
            const poll = setInterval(async () => {
                try {
                    const statusApiUrl = `${API_BASE_URL}/collect/tasks/${taskId}`;
                    const statusResponse = await fetch(statusApiUrl);
                    const statusData = await statusResponse.json();

                    // Update progress text for the user
                    if (statusData.progress?.phase) {
                        loadingStatusText.textContent = `Status: ${statusData.status} - ${statusData.progress.phase}...`;
                    }

                    if (statusData.status === 'completed') {
                        clearInterval(poll);
                        // In a real app, the 'result' would contain the aggregated profile.
                        // For now, we will simulate this by taking the raw collection stats.
                        // TODO: Replace this with a call to a new endpoint: GET /api/v1/creators/{channelId}/profile
                        populateUI(statusData.result);
                        showMainContent();
                    } else if (statusData.status === 'failed') {
                        clearInterval(poll);
                        throw new Error(statusData.error || 'Analysis failed.');
                    }
                } catch (pollError) {
                    clearInterval(poll);
                    showError(pollError.message);
                }
            }, 5000); // Poll every 5 seconds

        } catch (err) {
            showError(err.message);
        }
    };

    // --- UI MANIPULATION ---

    const populateUI = (data) => {
        // This function takes the final, aggregated data and populates the page.
        // For this POC, we'll use the placeholder data from the collection result.
        // In the future, this `data` object will be the rich aggregated profile.

        ui.setName(data.creator_name || 'Creator Name');
        ui.setHandle(data.channel_id); // Placeholder

        // Note: These stats will come from the aggregated profile in the future.
        ui.setStatFollowers(formatNumber(10500000)); // Placeholder
        ui.setStatAvgViews(formatNumber(2100000)); // Placeholder
        ui.setStatEngagement('4.5%'); // Placeholder
        ui.setStatVideos(formatNumber(1204)); // Placeholder

        ui.setQuickStatViews(formatNumber(2100000000)); // Placeholder
        ui.setQuickStatSubs(formatNumber(10500000)); // Placeholder
        ui.setQuickStatLikes(formatNumber(850000)); // Placeholder
        ui.setQuickStatComments(formatNumber(12300)); // Placeholder
    };

    const showMainContent = () => {
        loadingState.classList.add('hidden');
        mainContent.classList.remove('hidden');
    };

    const showError = (message) => {
        loadingState.innerHTML = `
            <div class="text-center text-red-400">
                <h3 class="text-xl font-bold">Analysis Failed</h3>
                <p class="mt-2">${message}</p>
            </div>
        `;
    };

    // --- INITIALIZATION ---
    const channelId = getChannelIdFromUrl();
    if (channelId) {
        // For now, we will just populate with dummy data to show the UI.
        // To enable live polling, uncomment the line below.
        populateUI({ creator_name: 'Loading...' }); // Show dummy name while loading
        showMainContent();

        // UNCOMMENT THIS LINE TO ENABLE THE LIVE API CALL AND POLLING
        // startCollectionAndPoll(channelId); 

    } else {
        showError("No Channel ID was provided in the URL. Please navigate from the discovery page.");
    }
});