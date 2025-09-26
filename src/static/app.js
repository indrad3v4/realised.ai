/**
 * Team-Realized PWA JavaScript
 * Handles property search, one-tap purchases, and user interactions
 */

// Global state
const AppState = {
    currentBudget: 500,
    currentCity: '',
    opportunities: [],
    loading: false,
    selectedProperty: null,
    isInitialized: false
};

// API endpoints
const API_BASE = '/api/v1';
const ENDPOINTS = {
    properties: `${API_BASE}/properties/affordable`,
    cityOpportunities: `${API_BASE}/cities`,
    propertyAnalysis: `${API_BASE}/analyze/property`,
    purchase: `${API_BASE}/purchase/one-tap`,
    health: `${API_BASE}/health`
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
async function initializeApp() {
    if (AppState.isInitialized) return;

    console.log('🚀 Initializing Team-Realized PWA');

    try {
        // Check API health
        const health = await checkAPIHealth();
        if (!health.ok) {
            showAlert('API temporarily unavailable. Some features may not work.', 'warning');
        }

        // Setup event listeners
        setupEventListeners();

        // Load saved preferences
        loadUserPreferences();

        // Setup PWA features
        setupPWA();

        AppState.isInitialized = true;
        console.log('✅ App initialized successfully');

    } catch (error) {
        console.error('❌ App initialization failed:', error);
        showAlert('App initialization failed. Please refresh the page.', 'error');
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Budget input changes
    const budgetInput = document.getElementById('budget');
    if (budgetInput) {
        budgetInput.addEventListener('input', debounce((e) => {
            AppState.currentBudget = parseFloat(e.target.value) || 500;
            saveUserPreferences();
        }, 300));
    }

    // City selection changes
    const citySelect = document.getElementById('city');
    if (citySelect) {
        citySelect.addEventListener('change', (e) => {
            AppState.currentCity = e.target.value;
            saveUserPreferences();
        });
    }

    // Bottom navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', handleNavigation);
    });

    // Modal close on outside click
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            closeModals();
        }
    });

    // Escape key to close modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModals();
        }
    });
}

/**
 * Main function to find investment opportunities
 */
async function findOpportunities() {
    if (AppState.loading) return;

    console.log(`🔍 Finding opportunities for €${AppState.currentBudget} budget`);

    try {
        // Update UI to show loading
        showLoading();

        // Build API URL
        const params = new URLSearchParams({
            budget: AppState.currentBudget,
            max_results: 8
        });

        if (AppState.currentCity) {
            params.append('location', AppState.currentCity);
        }

        // Make API request
        const response = await fetch(`${ENDPOINTS.properties}?${params}`);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        // Update state
        AppState.opportunities = data.opportunities || [];

        // Update UI
        hideLoading();
        displayOpportunities(data);

        // Analytics
        trackEvent('opportunities_found', {
            budget: AppState.currentBudget,
            city: AppState.currentCity,
            count: AppState.opportunities.length
        });

    } catch (error) {
        console.error('❌ Failed to find opportunities:', error);
        hideLoading();
        showAlert('Failed to find opportunities. Please try again.', 'error');
    }
}

/**
 * Show loading animation with fake progress
 */
function showLoading() {
    AppState.loading = true;

    const loadingSection = document.getElementById('loading');
    const resultsSection = document.getElementById('results');

    if (loadingSection) loadingSection.style.display = 'block';
    if (resultsSection) resultsSection.innerHTML = '';

    // Simulate scanning progress
    let citiesScanned = 0;
    let propertiesAnalyzed = 0;

    const progressInterval = setInterval(() => {
        if (!AppState.loading) {
            clearInterval(progressInterval);
            return;
        }

        citiesScanned = Math.min(citiesScanned + Math.floor(Math.random() * 3) + 1, 25);
        propertiesAnalyzed = Math.min(propertiesAnalyzed + Math.floor(Math.random() * 8) + 2, 120);

        updateElement('cities-scanned', citiesScanned);
        updateElement('properties-analyzed', propertiesAnalyzed);

        if (citiesScanned >= 25 && propertiesAnalyzed >= 120) {
            clearInterval(progressInterval);
        }
    }, 200);
}

/**
 * Hide loading animation
 */
function hideLoading() {
    AppState.loading = false;

    const loadingSection = document.getElementById('loading');
    if (loadingSection) {
        loadingSection.style.display = 'none';
    }
}

/**
 * Display found opportunities
 */
function displayOpportunities(data) {
    const resultsSection = document.getElementById('results');
    if (!resultsSection) return;

    if (!data.opportunities || data.opportunities.length === 0) {
        resultsSection.innerHTML = `
            <div class="alert alert-warning">
                <h4>No opportunities found</h4>
                <p>Try increasing your budget or selecting a different city.</p>
            </div>
        `;
        return;
    }

    // Create summary card
    const summaryHTML = `
        <div class="search-card fade-in">
            <h3>🎯 Found ${data.total_count} Investment Opportunities</h3>
            <p>Scanned ${data.cities_scanned} cities in ${data.analysis_time_seconds.toFixed(1)} seconds</p>
            <div class="summary-stats">
                <div class="summary-stat">
                    <span class="stat-number">${data.opportunities.filter(o => o.is_undervalued).length}</span>
                    <span class="stat-label">Undervalued</span>
                </div>
                <div class="summary-stat">
                    <span class="stat-number">€${data.user_budget}</span>
                    <span class="stat-label">Budget</span>
                </div>
                <div class="summary-stat">
                    <span class="stat-number">${data.opportunities.filter(o => o.attractiveness_score > 70).length}</span>
                    <span class="stat-label">High Score</span>
                </div>
            </div>
        </div>
    `;

    // Create opportunity cards
    const opportunitiesHTML = data.opportunities.map(createOpportunityCard).join('');

    resultsSection.innerHTML = summaryHTML + opportunitiesHTML;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Create HTML for a single opportunity card
 */
function createOpportunityCard(opportunity) {
    const badgeClass = opportunity.is_undervalued ? 'undervalued' : '';
    const badgeText = opportunity.is_undervalued 
        ? `${opportunity.undervaluation_percentage.toFixed(1)}% Undervalued`
        : 'Fair Priced';

    const ownershipPercent = ((opportunity.min_investment_amount / opportunity.total_property_value) * 100).toFixed(4);

    return `
        <div class="property-card fade-in" data-property-id="${opportunity.property_id}">
            <div class="property-header">
                <div class="property-title">
                    <h3>${opportunity.city} Property</h3>
                    <div class="property-address">${opportunity.address}</div>
                </div>
                <div class="property-badge ${badgeClass}">${badgeText}</div>
            </div>

            <div class="property-details">
                <div class="property-detail">
                    <span class="detail-label">Property Value:</span>
                    <span class="detail-value">€${formatNumber(opportunity.total_property_value)}</span>
                </div>
                <div class="property-detail">
                    <span class="detail-label">AI Fair Value:</span>
                    <span class="detail-value">€${formatNumber(opportunity.predicted_fair_value)}</span>
                </div>
                <div class="property-detail">
                    <span class="detail-label">Investment Score:</span>
                    <span class="detail-value">${opportunity.attractiveness_score.toFixed(0)}/100</span>
                </div>
                <div class="property-detail">
                    <span class="detail-label">Size:</span>
                    <span class="detail-value">${opportunity.size_sqm || 'N/A'} m²</span>
                </div>
            </div>

            <div class="property-price">
                <div class="investment-range">
                    Invest: €${opportunity.min_investment_amount} - €${opportunity.max_investment_amount}
                </div>
                <div class="ownership-info">
                    Own ${ownershipPercent}% of this property
                </div>
            </div>

            <div class="property-actions">
                <button class="btn btn-outline" onclick="analyzeProperty('${opportunity.property_id}')">
                    🧠 AI Analysis
                </button>
                <button class="btn btn-secondary" onclick="showPurchaseModal('${opportunity.property_id}', ${opportunity.min_investment_amount})">
                    💰 Buy €${opportunity.min_investment_amount} Piece
                </button>
            </div>
        </div>
    `;
}

/**
 * Show purchase confirmation modal
 */
function showPurchaseModal(propertyId, amount) {
    const opportunity = AppState.opportunities.find(o => o.property_id === propertyId);
    if (!opportunity) return;

    AppState.selectedProperty = { ...opportunity, selectedAmount: amount };

    const ownershipPercent = ((amount / opportunity.total_property_value) * 100).toFixed(4);
    const fees = amount * 0.03;
    const totalCost = amount + fees;

    const modalContent = `
        <div class="purchase-summary">
            <h4>${opportunity.city} Property Investment</h4>
            <p class="property-address">${opportunity.address}</p>
        </div>

        <div class="purchase-breakdown">
            <div class="breakdown-item">
                <span>Investment Amount:</span>
                <span>€${amount}</span>
            </div>
            <div class="breakdown-item">
                <span>Platform Fee (3%):</span>
                <span>€${fees.toFixed(2)}</span>
            </div>
            <div class="breakdown-item total">
                <span><strong>Total Cost:</strong></span>
                <span><strong>€${totalCost.toFixed(2)}</strong></span>
            </div>
            <div class="breakdown-item ownership">
                <span>You will own:</span>
                <span><strong>${ownershipPercent}%</strong></span>
            </div>
        </div>

        <div class="purchase-info">
            <p><small>You will receive an instant blockchain certificate after purchase.</small></p>
        </div>
    `;

    document.getElementById('purchase-details').innerHTML = modalContent;
    document.getElementById('purchase-modal').style.display = 'flex';

    // Track modal view
    trackEvent('purchase_modal_viewed', {
        property_id: propertyId,
        amount: amount,
        city: opportunity.city
    });
}

/**
 * Confirm and execute purchase
 */
async function confirmPurchase() {
    if (!AppState.selectedProperty) return;

    try {
        // Show loading state
        const confirmBtn = document.querySelector('#purchase-modal .btn-primary');
        const originalText = confirmBtn.innerHTML;
        confirmBtn.innerHTML = '<span class="spinner" style="width: 20px; height: 20px;"></span> Processing...';
        confirmBtn.disabled = true;

        // Create purchase request
        const purchaseRequest = {
            property_id: AppState.selectedProperty.property_id,
            investment_amount: AppState.selectedProperty.selectedAmount,
            user_id: getUserId(),
            payment_method: 'CARD'
        };

        // Make purchase API call
        const response = await fetch(ENDPOINTS.purchase, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(purchaseRequest)
        });

        const result = await response.json();

        if (result.success) {
            // Close purchase modal
            closePurchaseModal();

            // Show success modal
            showSuccessModal(result);

            // Track successful purchase
            trackEvent('purchase_completed', {
                property_id: AppState.selectedProperty.property_id,
                amount: AppState.selectedProperty.selectedAmount,
                transaction_id: result.transaction_id
            });

        } else {
            throw new Error(result.message || 'Purchase failed');
        }

    } catch (error) {
        console.error('❌ Purchase failed:', error);
        showAlert(`Purchase failed: ${error.message}`, 'error');

        // Reset button
        const confirmBtn = document.querySelector('#purchase-modal .btn-primary');
        confirmBtn.innerHTML = '<span class="btn-icon">💰</span> Confirm Purchase';
        confirmBtn.disabled = false;
    }
}

/**
 * Show success modal after purchase
 */
function showSuccessModal(result) {
    const successContent = `
        <div class="success-summary">
            <h4>Congratulations!</h4>
            <p>You now own ${result.ownership_percentage.toFixed(4)}% of this property.</p>
        </div>

        <div class="certificate-info">
            <div class="certificate-item">
                <span>Certificate ID:</span>
                <span class="certificate-id">${result.certificate_token}</span>
            </div>
            <div class="certificate-item">
                <span>Transaction ID:</span>
                <span class="transaction-id">${result.transaction_id}</span>
            </div>
            <div class="certificate-item">
                <span>Blockchain:</span>
                <span>${result.blockchain_network || 'Solana'}</span>
            </div>
            <div class="certificate-item">
                <span>Investment Amount:</span>
                <span>€${result.investment_amount || AppState.selectedProperty.selectedAmount}</span>
            </div>
        </div>

        <div class="next-steps">
            <p><small>Your ownership certificate has been stored on the blockchain. You can view and manage your portfolio in the Portfolio section.</small></p>
        </div>
    `;

    document.getElementById('success-details').innerHTML = successContent;
    document.getElementById('success-modal').style.display = 'flex';
}

/**
 * Analyze property with AI
 */
async function analyzeProperty(propertyId) {
    try {
        console.log(`🧠 Analyzing property: ${propertyId}`);

        const response = await fetch(`${ENDPOINTS.propertyAnalysis}/${propertyId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const analysis = await response.json();

        if (analysis) {
            showAnalysisModal(analysis);
            trackEvent('property_analyzed', { property_id: propertyId });
        }

    } catch (error) {
        console.error('❌ Property analysis failed:', error);
        showAlert('Property analysis failed. Please try again.', 'error');
    }
}

/**
 * Utility Functions
 */

function closeModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.style.display = 'none';
    });
}

function closePurchaseModal() {
    document.getElementById('purchase-modal').style.display = 'none';
    AppState.selectedProperty = null;
}

function closeSuccessModal() {
    document.getElementById('success-modal').style.display = 'none';
    // Navigate to portfolio (mock)
    showAlert('Portfolio feature coming soon!', 'info');
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(Math.round(num));
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} fade-in`;
    alert.textContent = message;

    // Insert at top of main content
    const main = document.querySelector('.app-main');
    if (main && main.firstChild) {
        main.insertBefore(alert, main.firstChild);
    }

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.parentNode.removeChild(alert);
        }
    }, 5000);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function getUserId() {
    // Simple user ID for demo - in production, use proper auth
    let userId = localStorage.getItem('team_realized_user_id');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('team_realized_user_id', userId);
    }
    return userId;
}

function saveUserPreferences() {
    const prefs = {
        budget: AppState.currentBudget,
        city: AppState.currentCity,
        timestamp: Date.now()
    };
    localStorage.setItem('team_realized_prefs', JSON.stringify(prefs));
}

function loadUserPreferences() {
    try {
        const prefs = JSON.parse(localStorage.getItem('team_realized_prefs') || '{}');
        if (prefs.budget) {
            AppState.currentBudget = prefs.budget;
            const budgetInput = document.getElementById('budget');
            if (budgetInput) budgetInput.value = prefs.budget;
        }
        if (prefs.city) {
            AppState.currentCity = prefs.city;
            const citySelect = document.getElementById('city');
            if (citySelect) citySelect.value = prefs.city;
        }
    } catch (error) {
        console.warn('Failed to load user preferences:', error);
    }
}

async function checkAPIHealth() {
    try {
        const response = await fetch('/health');
        return { ok: response.ok, status: response.status };
    } catch (error) {
        return { ok: false, error: error.message };
    }
}

function setupPWA() {
    // Service Worker registration (if available)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/sw.js')
            .then(reg => console.log('✅ Service Worker registered'))
            .catch(err => console.log('❌ Service Worker registration failed'));
    }

    // Install prompt handling
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        // Store for later use
        window.deferredPrompt = e;
    });
}

function handleNavigation(e) {
    e.preventDefault();

    // Remove active class from all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });

    // Add active class to clicked item
    e.currentTarget.classList.add('active');

    // Handle navigation based on href
    const href = e.currentTarget.getAttribute('href');

    switch(href) {
        case '#portfolio':
            showAlert('Portfolio feature coming soon!', 'info');
            break;
        case '#analyze':
            showAlert('AI Analysis feature coming soon!', 'info');
            break;
        case '#profile':
            showAlert('Profile feature coming soon!', 'info');
            break;
        default:
            // Home/Properties - scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function trackEvent(eventName, properties = {}) {
    // Analytics tracking - implement with your preferred analytics service
    console.log('📊 Event:', eventName, properties);

    // Example: Google Analytics 4
    // gtag('event', eventName, properties);

    // Example: Mixpanel
    // mixpanel.track(eventName, properties);
}

// Export for debugging
window.TeamRealized = {
    AppState,
    findOpportunities,
    showPurchaseModal,
    confirmPurchase,
    analyzeProperty
};

console.log('📱 Team-Realized PWA loaded successfully');
