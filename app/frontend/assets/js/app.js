document.addEventListener('DOMContentLoaded', () => {

    console.log("Step 1: Page loaded, script is running.");

    // --- CONFIGURATION ---
    const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

    // --- ELEMENT SELECTORS ---
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const creatorGrid = document.getElementById('creator-grid');
    const emptyState = document.getElementById('empty-state');
    const resultsSection = document.getElementById('results-section');

    console.log("Step 2: Found essential HTML elements:", { searchInput, searchButton, creatorGrid, emptyState });

    // --- STATE ---
    let isLoading = false;

    // --- CORE API FUNCTIONS ---

    const searchCreators = async (query) => {
        if (!query || isLoading) return;

        console.log(`Step 3: Starting search for query: "${query}"`);
        setLoadingState(true);

        const requestBody = {
            query: query,
            search_type: "creator",  // Changed from "topic" to "creator" - searches by name/handle
            filters: { platforms: ["YouTube"] },
            limit: 12
        };

        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            console.log("Step 4: Received response from API with status:", response.status);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An API error occurred.');
            }

            const data = await response.json();
            console.log("Step 5: Successfully parsed JSON data:", data);
            console.log("Step 5b: First result structure:", data.results?.[0]);
            console.log("Step 5c: Total results count:", data.results?.length);

            displayResults(data.results);

        } catch (error) {
            console.error("Search API Error:", error);
            displayError(error.message);
        } finally {
            setLoadingState(false);
        }
    };

    // TEMPORARY MOCK DATA GENERATOR (remove when backend has real data)
    const generateMockData = (query) => {
        const mockCreators = [
            {
                profile: {
                    name: `${query} Creator 1`,
                    handle: "@creator1",
                    avatar_url: "https://i.pravatar.cc/150?img=1",
                    platform_id: "UC_mock_1",
                    social_metrics: { followers_count: 1500000 },
                    engagement_metrics: { engagement_rate: 4.8 },
                    metadata: { categories: [query, "Tutorial", "Reviews"] }
                },
                match_confidence: 0.95
            },
            {
                profile: {
                    name: `${query} Expert`,
                    handle: "@expert",
                    avatar_url: "https://i.pravatar.cc/150?img=2",
                    platform_id: "UC_mock_2",
                    social_metrics: { followers_count: 850000 },
                    engagement_metrics: { engagement_rate: 5.2 },
                    metadata: { categories: [query, "Education", "Tips"] }
                },
                match_confidence: 0.88
            },
            {
                profile: {
                    name: `Pro ${query} Channel`,
                    handle: "@prochannel",
                    avatar_url: "https://i.pravatar.cc/150?img=3",
                    platform_id: "UC_mock_3",
                    social_metrics: { followers_count: 3200000 },
                    engagement_metrics: { engagement_rate: 3.9 },
                    metadata: { categories: [query, "Entertainment", "News"] }
                },
                match_confidence: 0.82
            }
        ];

        console.log("📦 Generated mock data:", mockCreators);
        return mockCreators;
    };

    // --- UI UPDATE FUNCTIONS ---

    const setLoadingState = (loading) => {
        isLoading = loading;
        searchButton.disabled = loading;
        searchButton.innerHTML = loading ? '<span class="truncate">Searching...</span>' : '<span class="truncate">Search</span>';

        resultsSection.classList.remove('hidden');
        emptyState.classList.add('hidden');

        creatorGrid.innerHTML = loading ? getSkeletonCardHTML(3) : '';
    };

    const displayResults = (results) => {
        console.log("Step 6: displayResults called with results:", results);

        // Show results section, hide empty state
        resultsSection.classList.remove('hidden');
        emptyState.classList.add('hidden');

        // Clear grid
        creatorGrid.innerHTML = '';

        if (!results || results.length === 0) {
            console.log("Step 7: No results found");
            creatorGrid.classList.add('hidden');
            emptyState.classList.remove('hidden');

            const emptyStateTitle = emptyState.querySelector('h3');
            const emptyStateMessage = emptyState.querySelector('p');
            if (emptyStateTitle) emptyStateTitle.textContent = 'No Results Found';
            if (emptyStateMessage) emptyStateMessage.textContent = 'Try adjusting your search terms or filters to find more creators.';
            return;
        }

        console.log(`Step 7: Found ${results.length} results. Generating cards...`);
        creatorGrid.classList.remove('hidden');

        // Force grid visibility with explicit height properties
        creatorGrid.style.cssText = `
            display: grid !important; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)) !important;
            grid-auto-rows: min-content !important;
            gap: 1.5rem !important; 
            visibility: visible !important; 
            opacity: 1 !important;
            min-height: 200px !important;
            width: 100% !important;
        `;

        // Create cards
        results.forEach((result, index) => {
            console.log(`Creating card ${index + 1}:`, result);
            const card = createCreatorCard(result);
            if (card) {
                creatorGrid.appendChild(card);
            }
        });

        console.log("Step 8: Finished rendering. Grid child count:", creatorGrid.children.length);

        // Auto-scroll to results after rendering
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    };

    const displayError = (message) => {
        resultsSection.classList.remove('hidden');
        creatorGrid.classList.add('hidden');
        creatorGrid.innerHTML = '';

        emptyState.classList.remove('hidden');
        emptyState.querySelector('h3').textContent = 'An Error Occurred';
        emptyState.querySelector('p').textContent = message;
    };

    // --- HTML TEMPLATE GENERATORS ---

    const createCreatorCard = (result) => {
        console.log("Creating card for result:", result);

        // Handle different possible API response structures
        let profile, matchScore;

        if (result.profile) {
            // Structure: { profile: {...}, match_confidence: 0.95 }
            profile = result.profile;
            matchScore = (result.match_confidence * 100).toFixed(0);
        } else if (result.name) {
            // Structure: { name: "...", handle: "...", ... } (flat structure)
            profile = result;
            matchScore = result.match_confidence ? (result.match_confidence * 100).toFixed(0) : 'N/A';
        } else {
            console.error("Unexpected result structure:", result);
            return null;
        }

        const social = profile.social_metrics || {};
        const engagementMetrics = profile.engagement_metrics || {};
        const metadata = profile.metadata || {};

        const name = profile.name || 'Unknown Creator';
        const handle = profile.handle || profile.channel_id || '@unknown';
        const avatar_url = profile.avatar_url || 'assets/images/placeholder.png';
        const followers = social.followers_count || social.subscriber_count || 0;
        const engagement = engagementMetrics.engagement_rate;
        const topics = metadata.categories || metadata.topics || [];
        const platform_id = profile.platform_id || profile.channel_id || profile.id;

        const formatNumber = (num) => {
            if (num === null || num === undefined || num === 0) return 'N/A';
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(0) + 'K';
            return num.toString();
        };

        const engagementText = (engagement !== null && engagement !== undefined)
            ? engagement.toFixed(1) + '%'
            : 'N/A';

        // Create card element with explicit visibility and layout
        const cardDiv = document.createElement('div');
        cardDiv.className = 'flex flex-col gap-4 rounded-xl p-6 glassmorphism';
        cardDiv.style.cssText = `
            display: flex !important; 
            visibility: visible !important; 
            opacity: 1 !important; 
            min-height: 350px !important;
            flex-direction: column !important;
            background: rgba(22, 25, 33, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        `;

        cardDiv.innerHTML = `
            <div class="flex items-center gap-4">
                <div class="relative">
                    <div class="rounded-full gradient-ring">
                        <img class="h-16 w-16 rounded-full object-cover" 
                             src="${avatar_url}" 
                             alt="${name}'s avatar"
                             onerror="this.src='assets/images/placeholder.png';"
                        />
                    </div>
                </div>
                <div class="flex-1 min-w-0">
                    <h3 class="text-xl font-bold text-[#F5F5F7] truncate">${name}</h3>
                    <p class="text-sm text-[#A1A1A6] truncate">${handle}</p>
                </div>
            </div>
            <div class="grid grid-cols-3 gap-4 text-center">
                <div>
                    <p class="text-2xl font-bold text-white">${formatNumber(followers)}</p>
                    <p class="text-xs text-[#A1A1A6]">Followers</p>
                </div>
                <div>
                    <p class="text-2xl font-bold text-white">${engagementText}</p>
                    <p class="text-xs text-[#A1A1A6]">Engagement</p>
                </div>
                <div>
                    <p class="text-2xl font-bold text-white">${matchScore}%</p>
                    <p class="text-xs text-[#A1A1A6]">Match</p>
                </div>
            </div>
            <div class="flex flex-wrap gap-2 min-h-[32px]">
                ${topics.slice(0, 3).map(topic =>
            `<span class="text-xs font-medium bg-primary/20 text-primary px-2.5 py-1 rounded-full whitespace-nowrap">
                        ${topic}
                    </span>`
        ).join('')}
            </div>
            <button class="mt-2 w-full flex items-center justify-center rounded-lg h-11 px-4 bg-white/5 text-white font-semibold hover:bg-white/10 transition-colors" 
                    data-channel-id="${platform_id}">
                View Deep Analysis
            </button>
        `;

        // Add click handler
        const button = cardDiv.querySelector('button');
        button.addEventListener('click', () => {
            handleDeepDive(platform_id);
        });

        return cardDiv;
    };

    const getSkeletonCardHTML = (count = 1) => {
        const cards = [];
        for (let i = 0; i < count; i++) {
            cards.push(`
                <div class="flex flex-col gap-4 rounded-xl p-6 glassmorphism relative overflow-hidden">
                    <div class="absolute inset-0 shimmer"></div>
                    <div class="flex items-center gap-4">
                        <div class="h-16 w-16 rounded-full bg-white/5"></div>
                        <div class="flex-1">
                            <div class="h-6 w-3/4 rounded bg-white/5 mb-2"></div>
                            <div class="h-4 w-1/2 rounded bg-white/5"></div>
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-4">
                        <div class="h-12 rounded bg-white/5"></div>
                        <div class="h-12 rounded bg-white/5"></div>
                        <div class="h-12 rounded bg-white/5"></div>
                    </div>
                    <div class="flex gap-2">
                        <div class="h-6 w-16 rounded-full bg-white/5"></div>
                        <div class="h-6 w-20 rounded-full bg-white/5"></div>
                    </div>
                    <div class="h-11 w-full rounded-lg bg-white/5 mt-2"></div>
                </div>
            `);
        }
        return cards.join('');
    };

    const handleDeepDive = (channelId) => {
        console.log("Navigating to deep dive page for channel:", channelId);
        window.location.href = `creator.html?channel_id=${channelId}`;
    };

    // --- EVENT LISTENERS ---
    searchButton.addEventListener('click', () => {
        const query = searchInput.value.trim();
        if (query) {
            searchCreators(query);
        }
    });

    searchInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                searchCreators(query);
            }
        }
    });

    // Initialize UI state
    const initializeUI = () => {
        resultsSection.classList.remove('hidden');
        emptyState.classList.remove('hidden');
        creatorGrid.classList.add('hidden');

        emptyState.querySelector('h3').textContent = 'Discover Your Next Creator';
        emptyState.querySelector('p').textContent = 'Use the search bar above to find creators by niche, topic, or name';

        console.log('UI initialized');
    };

    initializeUI();

    // Make deep dive handler globally accessible
    window.startDeepDive = handleDeepDive;
});