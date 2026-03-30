// Base API URL
const API_BASE_URL = 'https://ekyc-backend-750223193485.asia-south2.run.app';

// Global Data Arrays
let verificationData = [];
let alertData = [];
let documentData = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', async function() {
    // Check which page we're on
    const path = window.location.pathname;
    
    // Fetch data from FastAPI Backend first
    try {
        await fetchDashboardData();
    } catch (error) {
        console.error("Failed to load data from backend:", error);
    }
    
    if (path.includes('dashboard.html') || path.endsWith('/dashboard')) {
        initDashboard();
    } else if (path.includes('verifications.html') || path.endsWith('/verifications')) {
        initVerifications();
    } else if (path.includes('alerts.html') || path.endsWith('/alerts')) {
        initAlerts();
    } else if (path.includes('documents.html') || path.endsWith('/documents')) {
        initDocuments();
    } else if (path.includes('analytics.html') || path.endsWith('/analytics')) {
        initAnalytics();
    } else if (path.includes('settings.html') || path.endsWith('/settings')) {
        initSettings();
    }
    
    // Update stats if on dashboard
    if (document.getElementById('totalApps')) {
        updateDashboardStats();
    }
    
    // Load current user
    loadCurrentUser();
});

// Fetch Data from Backend
async function fetchDashboardData() {
    try {
        const [verRes, alertRes, docRes] = await Promise.all([
            fetch(`${API_BASE_URL}/verifications`),
            fetch(`${API_BASE_URL}/alerts`),
            fetch(`${API_BASE_URL}/documents`)
        ]);

        if (verRes.ok) {
            const vPayload = await verRes.json();
            const vData = Array.isArray(vPayload) ? vPayload : (vPayload.verifications || []);
            // Map backend schema to frontend expected formats
            verificationData = vData.map(v => ({
                id: v.id,
                name: 'User ' + v.user_id, // Placeholder until users are joined
                email: `user${v.user_id}@email.com`,
                docType: 'Document', 
                date: new Date(v.created_at).toISOString().split('T')[0],
                status: v.status.toLowerCase(),
                riskScore: v.risk_score
            }));
        }

        if (alertRes.ok) {
            const aPayload = await alertRes.json();
            const aData = Array.isArray(aPayload) ? aPayload : (aPayload.alerts || []);
            alertData = aData.map(a => ({
                id: a.id,
                name: 'User ' + (a.user_id || 'N/A'),
                risk: a.risk_level,
                type: a.alert_type,
                date: new Date(a.created_at).toISOString().split('T')[0],
                status: a.status
            }));
        }

        if (docRes.ok) {
            const dPayload = await docRes.json();
            const dData = Array.isArray(dPayload) ? dPayload : (dPayload.documents || []);
            documentData = dData.map(d => ({
                id: d.id,
                type: d.type || 'Document',
                name: 'User ' + d.user_id,
                date: new Date(d.created_at).toISOString().split('T')[0],
                status: (d.status || 'processed').toLowerCase()
            }));
        }
    } catch (e) {
        console.error("Error fetching data:", e);
    }
}


// Load current user from localStorage
function loadCurrentUser() {
    let currentUser = JSON.parse(localStorage.getItem('currentUser'));
    
    if (currentUser) {
        // Update profile elements across all pages
        const profileNameElements = document.querySelectorAll('#profileName, .user-name');
        const profileRoleElements = document.querySelectorAll('#profileRole, .user-role');
        const profileAvatarElements = document.querySelectorAll('#profileAvatar, .user-avatar, #headerProfileBtn');
        
        profileNameElements.forEach(el => {
            if (el) el.textContent = currentUser.name;
        });
        
        profileRoleElements.forEach(el => {
            if (el) el.textContent = currentUser.role || 'Compliance Officer';
        });
        
        const initials = currentUser.name.split(' ').map(n => n[0]).join('').toUpperCase();
        profileAvatarElements.forEach(el => {
            if (el) el.textContent = initials;
        });
    }
}

// Switch between login and register tabs
function switchAuthTab(tab) {
    const loginTab = document.querySelectorAll('.login-tab');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    
    if (tab === 'login') {
        loginTab[0].classList.add('active');
        loginTab[1].classList.remove('active');
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
    } else {
        loginTab[0].classList.remove('active');
        loginTab[1].classList.add('active');
        loginForm.classList.remove('active');
        registerForm.classList.add('active');
    }
}

// Login function
function login() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    if (email && password) {
        // Show loading animation
        const btn = document.querySelector('.login-btn');
        btn.innerHTML = 'Signing in...';
        btn.disabled = true;
        
        // Simulate API call
        setTimeout(() => {
            window.location.href = 'dashboard.html';
        }, 1000);
    } else {
        showNotification('Please enter both email and password', 'error');
    }
}

// Register function
function register() {
    const name = document.getElementById('reg-name')?.value;
    const email = document.getElementById('reg-email')?.value;
    const password = document.getElementById('reg-password')?.value;
    const confirm = document.getElementById('reg-confirm')?.value;
    
    if (!name || !email || !password || !confirm) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    
    if (password !== confirm) {
        showNotification('Passwords do not match', 'error');
        return;
    }
    
    // Show loading animation
    const btn = document.querySelector('.login-btn');
    btn.innerHTML = 'Creating account...';
    btn.disabled = true;
    
    // Simulate API call
    setTimeout(() => {
        showNotification('Account created successfully! Please login.', 'success');
        switchAuthTab('login');
        btn.innerHTML = 'Create Account';
        btn.disabled = false;
    }, 1000);
}

// Initialize Dashboard
function initDashboard() {
    populateKYCTable();
    updateDashboardStats();
    
    // Set up auto-refresh every 30 seconds
    setInterval(() => {
        updateDashboardStats();
    }, 30000);
}

// Update dashboard statistics
function updateDashboardStats() {
    const total = verificationData.length + 12000; // Base + sample
    const verified = verificationData.filter(d => d.status === 'verified').length + 9800;
    const flagged = verificationData.filter(d => d.status === 'flagged').length + 235;
    const pending = verificationData.filter(d => d.status === 'pending').length + 2700;
    
    // Update stats with animation
    animateValue('totalApps', 12000, total, 1000);
    animateValue('verifiedCount', 9800, verified, 1000);
    animateValue('flaggedCount', 235, flagged, 1000);
    animateValue('pendingCount', 2700, pending, 1000);
}

// Animate number changes
function animateValue(elementId, start, end, duration) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current).toLocaleString();
    }, 16);
}

// Populate KYC table on dashboard
function populateKYCTable() {
    const tbody = document.getElementById('kycTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    verificationData.slice(0, 5).forEach(item => {
        const row = document.createElement('tr');
        
        const statusClass = item.status === 'verified' ? 'verified' : 
                           (item.status === 'flagged' ? 'flagged' : 'pending');
        
        const riskClass = item.riskScore > 70 ? 'high' : 
                         (item.riskScore > 30 ? 'medium' : 'low');
        
        row.innerHTML = `
            <td>${item.id}</td>
            <td>${item.name}</td>
            <td>${item.docType}</td>
            <td>${item.date}</td>
            <td><span class="status-badge ${statusClass}">${item.status.toUpperCase()}</span></td>
            <td><span class="risk-score ${riskClass}">${item.riskScore}%</span></td>
            <td><button class="view-btn" onclick="viewDetails('${item.id}')">View</button></td>
        `;
        
        tbody.appendChild(row);
    });
}

// Initialize Verifications page
function initVerifications() {
    populateVerificationsTable();
}

// Populate verifications table
function populateVerificationsTable(filterStatus = 'all', filterRisk = 'all', searchTerm = '') {
    const tbody = document.getElementById('verificationsTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    let filteredData = [...verificationData];
    
    // Apply status filter
    if (filterStatus !== 'all') {
        filteredData = filteredData.filter(item => item.status === filterStatus);
    }
    
    // Apply risk filter
    if (filterRisk !== 'all') {
        if (filterRisk === 'low') {
            filteredData = filteredData.filter(item => item.riskScore <= 30);
        } else if (filterRisk === 'medium') {
            filteredData = filteredData.filter(item => item.riskScore > 30 && item.riskScore <= 70);
        } else if (filterRisk === 'high') {
            filteredData = filteredData.filter(item => item.riskScore > 70);
        }
    }
    
    // Apply search
    if (searchTerm) {
        const term = searchTerm.toLowerCase();
        filteredData = filteredData.filter(item => 
            item.name.toLowerCase().includes(term) || 
            item.email.toLowerCase().includes(term) ||
            item.id.toLowerCase().includes(term)
        );
    }
    
    if (filteredData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 30px;">No verifications found</td></tr>';
        return;
    }
    
    filteredData.forEach(item => {
        const row = document.createElement('tr');
        
        const statusClass = item.status === 'verified' ? 'verified' : 
                           (item.status === 'flagged' ? 'flagged' : 'pending');
        
        const riskClass = item.riskScore > 70 ? 'high' : 
                         (item.riskScore > 30 ? 'medium' : 'low');
        
        row.innerHTML = `
            <td>${item.id}</td>
            <td>${item.name}</td>
            <td>${item.email}</td>
            <td>${item.docType}</td>
            <td>${item.date}</td>
            <td><span class="status-badge ${statusClass}">${item.status.toUpperCase()}</span></td>
            <td><span class="risk-score ${riskClass}">${item.riskScore}%</span></td>
            <td>
                <button class="view-btn" onclick="viewDetails('${item.id}')" style="margin-right: 5px;">View</button>
                <button class="view-btn" style="background: var(--danger); color: white; border: none;" onclick="flagItem('${item.id}')">Flag</button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Search verifications
function searchVerifications() {
    const searchTerm = document.getElementById('searchInput')?.value || '';
    const statusFilter = document.getElementById('statusFilter')?.value || 'all';
    const riskFilter = document.getElementById('riskFilter')?.value || 'all';
    
    populateVerificationsTable(statusFilter, riskFilter, searchTerm);
}

// Filter verifications
function filterVerifications() {
    const searchTerm = document.getElementById('searchInput')?.value || '';
    const statusFilter = document.getElementById('statusFilter')?.value || 'all';
    const riskFilter = document.getElementById('riskFilter')?.value || 'all';
    
    populateVerificationsTable(statusFilter, riskFilter, searchTerm);
}

// Initialize Alerts page
function initAlerts() {
    populateAlertsTable();
}

// Populate alerts table
function populateAlertsTable() {
    const tbody = document.getElementById('alertsTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    alertData.forEach(alert => {
        const row = document.createElement('tr');
        
        const riskClass = alert.risk === 'High' ? 'danger' : 
                         (alert.risk === 'Medium' ? 'warning' : 'success');
        
        row.innerHTML = `
            <td>${alert.id}</td>
            <td>${alert.name}</td>
            <td><span class="status-badge ${riskClass}">${alert.risk}</span></td>
            <td>${alert.type}</td>
            <td>${alert.date}</td>
            <td><span class="status-badge">${alert.status}</span></td>
            <td>
                <button class="view-btn" onclick="resolveAlert('${alert.id}')" style="margin-right: 5px;">Resolve</button>
                <button class="view-btn" style="background: var(--danger); color: white; border: none;" onclick="investigateAlert('${alert.id}')">Investigate</button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Initialize Documents page
function initDocuments() {
    populateDocumentsTable();
}

// Populate documents table
function populateDocumentsTable() {
    const tbody = document.getElementById('documentsTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    documentData.forEach(doc => {
        const row = document.createElement('tr');
        
        const statusClass = doc.status === 'verified' ? 'verified' : 
                           (doc.status === 'flagged' ? 'flagged' : 'pending');
        
        row.innerHTML = `
            <td>${doc.id}</td>
            <td>${doc.type}</td>
            <td>${doc.name}</td>
            <td>${doc.date}</td>
            <td><span class="status-badge ${statusClass}">${doc.status.toUpperCase()}</span></td>
            <td>
                <button class="view-btn" onclick="viewDocument('${doc.id}')">View</button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// Initialize Analytics page
function initAnalytics() {
    // Setup time range buttons
    const rangeBtns = document.querySelectorAll('.range-btn');
    rangeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            rangeBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            updateAnalyticsCharts(this.textContent);
        });
    });
}

// Update analytics charts
function updateAnalyticsCharts(range) {
    console.log(`Updating charts for range: ${range}`);
    showNotification(`Analytics updated for ${range}`, 'info');
}

// Initialize Settings page
function initSettings() {
    // Show general settings by default
    showSettingsTab('general');
}

// Show settings tab
function showSettingsTab(tabName) {
    // Update tab buttons
    const tabs = document.querySelectorAll('.settings-tab');
    tabs.forEach(tab => {
        const tabText = tab.textContent.toLowerCase();
        if (tabText.includes(tabName)) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    
    // Show corresponding panel
    const panels = document.querySelectorAll('.settings-panel');
    panels.forEach(panel => {
        panel.classList.remove('active');
    });
    
    const activePanel = document.getElementById(`${tabName}-settings`);
    if (activePanel) {
        activePanel.classList.add('active');
    }
}

// Save settings
function saveSettings() {
    showNotification('Settings saved successfully!', 'success');
}

// Copy to clipboard
function copyToClipboard() {
    const apiKeyInput = document.querySelector('.api-key-display input');
    if (apiKeyInput) {
        apiKeyInput.select();
        document.execCommand('copy');
        showNotification('API key copied to clipboard!', 'success');
    }
}

// Trigger file upload
function triggerUpload() {
    document.getElementById('fileInput').click();
}

// Handle file upload
function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    
    const file = files[0];
    const reader = new FileReader();
    
    reader.onload = function(e) {
        try {
            if (file.name.endsWith('.json')) {
                const newData = JSON.parse(e.target.result);
                
                // Add new data to appropriate arrays
                if (Array.isArray(newData)) {
                    verificationData = [...verificationData, ...newData];
                    showNotification(`Uploaded ${newData.length} records successfully!`, 'success');
                    
                    // Refresh current page
                    if (window.location.pathname.includes('verifications')) {
                        populateVerificationsTable();
                    } else if (window.location.pathname.includes('dashboard')) {
                        populateKYCTable();
                        updateDashboardStats();
                    }
                }
            } else if (file.name.endsWith('.csv')) {
                // Simple CSV parsing
                const lines = e.target.result.split('\n');
                const headers = lines[0].split(',').map(h => h.trim());
                
                const newRecords = [];
                for (let i = 1; i < lines.length; i++) {
                    if (lines[i].trim()) {
                        const values = lines[i].split(',').map(v => v.trim());
                        const record = {};
                        headers.forEach((header, index) => {
                            record[header] = values[index] || '';
                        });
                        newRecords.push(record);
                    }
                }
                
                verificationData = [...verificationData, ...newRecords];
                showNotification(`Uploaded ${newRecords.length} records from CSV!`, 'success');
            } else {
                showNotification('Please upload a JSON or CSV file', 'error');
            }
        } catch (error) {
            showNotification('Error parsing file. Please check the format.', 'error');
            console.error(error);
        }
    };
    
    if (file.name.endsWith('.json') || file.name.endsWith('.csv')) {
        reader.readAsText(file);
    } else {
        showNotification('Please upload a JSON or CSV file', 'error');
    }
}

// Upload front image
function uploadFrontImage() {
    showNotification('Front image upload feature coming soon!', 'info');
}

// Upload back image
function uploadBackImage() {
    showNotification('Back image upload feature coming soon!', 'info');
}

// View details
function viewDetails(id) {
    showNotification(`Viewing details for ${id}`, 'info');
}

// View document
function viewDocument(id) {
    showNotification(`Viewing document ${id}`, 'info');
}

// Flag item
function flagItem(id) {
    const item = verificationData.find(i => i.id === id);
    if (item) {
        item.status = 'flagged';
        item.riskScore = Math.min(100, item.riskScore + 50);
        filterVerifications();
        showNotification(`Item ${id} has been flagged for review`, 'warning');
    }
}

// Resolve alert
function resolveAlert(alertId) {
    const alert = alertData.find(a => a.id === alertId);
    if (alert) {
        alert.status = 'Resolved';
        populateAlertsTable();
        showNotification(`Alert ${alertId} resolved successfully!`, 'success');
    }
}

// Investigate alert
function investigateAlert(alertId) {
    showNotification(`Investigating alert ${alertId}...`, 'info');
}

// Start new review
function startNewReview() {
    showNotification('Starting new review session', 'info');
}

// Export data
function exportData() {
    const dataStr = JSON.stringify(verificationData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `kyc-data-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    showNotification('Data exported successfully!', 'success');
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notification
    const existing = document.querySelector('.custom-notification');
    if (existing) existing.remove();
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'custom-notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        border-radius: var(--radius, 12px);
        color: white;
        font-weight: 500;
        box-shadow: var(--shadow-lg, 0 10px 15px rgba(0,0,0,0.1));
        z-index: 1000;
        animation: slideIn 0.3s ease;
        cursor: pointer;
    `;
    
    // Set background color based on type
    if (type === 'success') {
        notification.style.background = '#06d6a0';
    } else if (type === 'error') {
        notification.style.background = '#ef476f';
    } else if (type === 'warning') {
        notification.style.background = '#ffd166';
        notification.style.color = '#212529';
    } else {
        notification.style.background = '#4361ee';
    }
    
    notification.textContent = message;
    
    // Add click to dismiss
    notification.addEventListener('click', function() {
        document.body.removeChild(notification);
    });
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (document.body.contains(notification)) {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }
    }, 3000);
}

// Add slide out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// Confirm fraud
function confirmFraud() {
    showNotification('Fraud confirmed and case escalated', 'warning');
}

// Clear case
function clearCase() {
    showNotification('Case cleared from review queue', 'success');
}

// Set time range for analytics
function setTimeRange(range) {
    const btns = document.querySelectorAll('.range-btn');
    btns.forEach(btn => {
        if (btn.textContent.includes(range)) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    updateAnalyticsCharts(range);
}

// Handle window resize
let resizeTimer;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function() {
        // Refresh charts if on analytics page
        if (window.location.pathname.includes('analytics')) {
            console.log('Window resized - updating charts');
        }
    }, 250);
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + R to refresh
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        if (window.location.pathname.includes('dashboard')) {
            updateDashboardStats();
            populateKYCTable();
            showNotification('Dashboard refreshed!', 'success');
        }
    }
    
    // Ctrl/Cmd + E to export
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        exportData();
    }
    
    // Escape to close notifications
    if (e.key === 'Escape') {
        const notifications = document.querySelectorAll('.custom-notification');
        notifications.forEach(notification => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        });
    }
});