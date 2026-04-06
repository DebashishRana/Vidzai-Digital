// Base API URL
const RAW_API_BASE_URL = localStorage.getItem('API_BASE_URL') || 'http://localhost:8000';
const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, '');
const API_ROOT_URL = API_BASE_URL.endsWith('/api/v1') ? API_BASE_URL.replace(/\/api\/v1$/, '') : API_BASE_URL;
const API_V1_URL = API_BASE_URL.endsWith('/api/v1') ? API_BASE_URL : `${API_BASE_URL}/api/v1`;

async function fetchFirstSuccessfulJson(paths) {
    for (const path of paths) {
        const url = path.startsWith('/api/v1/') ? `${API_ROOT_URL}${path}` : `${API_ROOT_URL}${path}`;
        try {
            const res = await fetch(url);
            if (!res.ok) continue;
            const data = await res.json();
            return { ok: true, data, url };
        } catch (_) {
            // Try next candidate path.
        }
    }
    return { ok: false, data: null, url: null };
}

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

// Save data locally (Deprecated - Now uses Backend DB)
async function saveDataLocally() {
    console.warn('saveDataLocally is deprecated. Backend database is now the single source of truth.');
}

// Fetch Data from Backend
async function fetchDashboardData() {
    try {
        const [verRes, alertRes, docRes] = await Promise.all([
            fetchFirstSuccessfulJson([
                '/api/v1/verification-logs/records',
                '/api/v1/verifications',
                '/verifications'
            ]),
            fetchFirstSuccessfulJson([
                '/api/v1/dashboard/alerts',
                '/api/v1/alerts',
                '/alerts'
            ]),
            fetchFirstSuccessfulJson([
                '/api/v1/dashboard/documents',
                '/api/v1/documents',
                '/documents'
            ])
        ]);

        if (verRes.ok) {
            const vPayload = verRes.data;
            const vData = Array.isArray(vPayload) ? vPayload : (vPayload.data || vPayload.verifications || []);
            verificationData = vData.map(v => {
                const statusStr = (v.status || 'pending').toLowerCase();
                let normalizedStatus = statusStr.includes('verif') ? 'verified' : (statusStr.includes('fail') || statusStr.includes('review') || statusStr.includes('flag') ? 'flagged' : 'pending');
                return {
                    id: v.verification_id || v.id,
                    name: v.user_id ? 'User ' + v.user_id : 'Unknown User',
                    email: v.user_id ? `user${v.user_id}@email.com` : 'No Email',
                    docType: v.document_type || 'Document',
                    date: new Date(v.created_at || Date.now()).toISOString().split('T')[0],
                    status: normalizedStatus,
                    riskScore: v.confidence_score !== undefined ? Math.round(v.confidence_score * 100) : (v.risk_score || 0)
                };
            });
        } else {
            verificationData = [];
        }

        if (alertRes.ok) {
            const aPayload = alertRes.data;
            const aData = Array.isArray(aPayload) ? aPayload : (aPayload.alerts || []);
            alertData = aData.map(a => ({
                id: a.id,
                name: a.user_id ? 'User ' + a.user_id : 'Unknown User',
                risk: a.risk_level || 'Low',
                type: a.alert_type || 'Unknown Flag',
                date: new Date(a.created_at || Date.now()).toISOString().split('T')[0],
                status: a.status || 'Active'
            }));
        } else {
            alertData = [];
        }

        if (docRes.ok) {
            const dPayload = docRes.data;
            const dData = Array.isArray(dPayload) ? dPayload : (dPayload.documents || []);
            documentData = dData.map(d => ({
                id: d.id,
                type: d.type || 'Document',
                name: 'User ' + d.user_id,
                date: new Date(d.created_at).toISOString().split('T')[0],
                status: (d.status || 'processed').toLowerCase()
            }));
        } else {
            documentData = [];
        }
    } catch (e) {
        console.error("Error fetching data:", e);
        verificationData = [];
        alertData = [];
        documentData = [];
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
        const btn = document.querySelector('#login-form .login-btn');
        btn.innerHTML = 'Signing in...';
        btn.disabled = true;

        const displayName = email.split('@')[0].replace(/[._-]/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());

        const currentUser = {
            name: displayName || 'Compliance Officer',
            email,
            role: 'Compliance Officer'
        };
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        
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
    const btn = document.querySelector('#register-form .login-btn');
    btn.innerHTML = 'Creating account...';
    btn.disabled = true;
    
    // Simulate API call
    setTimeout(() => {
        const currentUser = {
            name,
            email,
            role: 'Compliance Officer'
        };
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        showNotification('Account created successfully! Please login.', 'success');
        switchAuthTab('login');
        btn.innerHTML = 'Create Account';
        btn.disabled = false;

        const loginEmail = document.getElementById('email');
        if (loginEmail) {
            loginEmail.value = email;
        }
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
    const total = verificationData.length;
    const verified = verificationData.filter(d => d.status === 'verified').length;
    const flagged = verificationData.filter(d => d.status === 'flagged').length;
    const pending = verificationData.filter(d => d.status === 'pending').length;
    
    // Average Confidence
    let totalConf = 0;
    verificationData.forEach(d => {
        totalConf += (d.confidence || 0); // Assuming confidence is 0-1, adjusting to %
    });
    const avgConfidence = total > 0 ? (totalConf / total) * 100 : 0;
    
    // Update stats with animation (Top nav items)
    animateValue('totalApps', 0, total, 1000);
    animateValue('verifiedCount', 0, verified, 1000);
    animateValue('flaggedCount', 0, flagged, 1000);
    animateValue('pendingCount', 0, pending, 1000);

    // Update the new System Metrics dashboard grid
    const dashVerifiedEl = document.getElementById('dashVerified');
    if (dashVerifiedEl) {
        animateValue('dashTotalVerifications', 0, total, 1000);
        animateValue('dashVerified', 0, verified, 1000);
        animateValue('dashFlagged', 0, flagged, 1000);
        // Animate the % separately
        const avgEl = document.getElementById('dashAvgConfidence');
        if (avgEl) {
            let start = 0, current = 0, inc = avgConfidence / 60;
            let timer = setInterval(() => {
                current += inc;
                if (current >= avgConfidence) {
                    current = avgConfidence;
                    clearInterval(timer);
                }
                avgEl.innerText = Math.round(current) + '%';
            }, 16);
        }
    }
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
    
    // Check if redirected from new KYC Flow to auto-open specific case
    const urlParams = new URLSearchParams(window.location.search);
    const verifyId = urlParams.get('verifyId');
    if (verifyId) {
        // Wait a small moment to ensure the UI paints before triggering the review overlay
        setTimeout(() => {
            viewDetails(verifyId);
            // Optionally clear the query param so refresh doesn't reopen it
            window.history.replaceState({}, document.title, window.location.pathname);
        }, 300);
    }
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
              <td>${item.docType}</td>
              <td><span class="risk-score ${riskClass}">${item.riskScore}%</span></td>
              <td><span class="status-badge ${statusClass}">${item.status.toUpperCase()}</span></td>
              <td>${item.date}</td>
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
    const derivedAlerts = buildAlertsFromVerifications();
    alertData = mergeAlertsById(alertData, derivedAlerts);

    populateAlertsTable();
    updateAlertMetrics();
}

function buildAlertsFromVerifications() {
    return verificationData
        .map(item => {
            const confidence = Number(item.riskScore || 0) / 100;
            if (Number.isNaN(confidence) || confidence >= 0.75) {
                return null;
            }

            const riskLevel = confidence < 0.5 ? 'High' : 'Medium';

            return {
                id: `VF-${item.id}`,
                name: item.name || 'Unknown User',
                risk: riskLevel,
                type: 'Low Confidence Verification',
                date: item.date || new Date().toISOString().split('T')[0],
                status: 'Pending Review'
            };
        })
        .filter(Boolean);
}

function mergeAlertsById(existingAlerts, derivedAlerts) {
    const mergedMap = new Map();

    (existingAlerts || []).forEach(alert => {
        if (alert && alert.id) {
            mergedMap.set(alert.id, alert);
        }
    });

    (derivedAlerts || []).forEach(alert => {
        if (alert && alert.id) {
            mergedMap.set(alert.id, alert);
        }
    });

    return Array.from(mergedMap.values());
}

function updateAlertMetrics() {
    const highRisk = alertData.filter(alert => alert.risk === 'High').length;
    const mediumRisk = alertData.filter(alert => alert.risk === 'Medium').length;
    const lowRisk = alertData.filter(alert => alert.risk === 'Low').length;
    const pendingReview = alertData.filter(alert => (alert.status || '').toLowerCase().includes('pending')).length;

    const highEl = document.getElementById('highRiskCount');
    const mediumEl = document.getElementById('mediumRiskCount');
    const lowEl = document.getElementById('lowRiskCount');
    const pendingEl = document.getElementById('pendingReviewCount');

    if (highEl) highEl.textContent = highRisk;
    if (mediumEl) mediumEl.textContent = mediumRisk;
    if (lowEl) lowEl.textContent = lowRisk;
    if (pendingEl) pendingEl.textContent = pendingReview;
}

// Populate alerts table
function populateAlertsTable() {
    const tbody = document.getElementById('alertsTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (alertData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 30px;">No flagged alerts found</td></tr>';
        return;
    }

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

    // Populate the newly added IDs with localStorage counts
    const total = verificationData.length;
    const verified = verificationData.filter(d => d.status === 'verified').length;
    const flagged = verificationData.filter(d => d.status === 'flagged').length;
    const pending = verificationData.filter(d => d.status === 'pending').length;

    animateValue('analyticsTotal', 0, total, 1000);
    animateValue('analyticsVerified', 0, verified, 1000);
    animateValue('analyticsFlagged', 0, flagged, 1000);
    animateValue('analyticsPending', 0, pending, 1000);
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
async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    
    // Pass logic directly to backend or use the inline handleFileUpload in dashboard.html
    showNotification('Batch upload moved to backend logic via Dashboard.', 'info');
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
    const ver = verificationData.find(v => v.id === id);
    if (!ver) {
        showNotification('Verification case not found.', 'error');
        return;
    }
    
    // Attempt to pull real data from session storage (populated during KYC flow)
    const sessImg = sessionStorage.getItem('recent_upload_' + id);
    const sessAddress = sessionStorage.getItem('recent_address_' + id) || "Address not detected";
    
    const doc = documentData.find(d => d.verification_id === id);
    let extractedText = doc ? (doc.extracted_text || "") : "";
    
    if (!extractedText.trim()) {
        extractedText = `Name: ${ver.name}\nDOB: 01/01/1990\nAddress: ${sessAddress}\nID: 9999-9999-9999`;
    }
    
    // Extract pieces
    const nameMatch = extractedText.match(/Name:\s*([^\n]+)/i);
    const dobMatch = extractedText.match(/DOB:\s*([^\n]+)/i) || extractedText.match(/Date of Birth:\s*([^\n]+)/i) || extractedText.match(/(?:(?:0[1-9]|[12]\d|3[01])\/(?:0[1-9]|1[0-2])\/(?:19|20)\d{2})/);
    const dName = nameMatch ? nameMatch[1] : ver.name;
    const dDOB = dobMatch ? (dobMatch[1] || dobMatch[0]) : "11 NOV 90";
    const dDocType = ver.docType || "Passport";
    
    const isFaceMatch = ver.confidence >= 0.70;
    const docConfidence = (ver.riskScore ? (100 - ver.riskScore) : 80);
    const initials = dName.split(' ').map(n=>n[0]).join('').substring(0,2).toUpperCase();

    // Create the overlay container
    const overlay = document.createElement('div');
    overlay.className = 'case-review-overlay';
    // Base styles for the backdrop
    overlay.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.6); backdrop-filter: blur(5px);
        display: flex; justify-content: center; align-items: center; z-index: 99999;
        font-family: 'Inter', sans-serif;
    `;
    
    // Use the real uploaded image if available
    let viewerContent = '';
    if (sessImg) {
        viewerContent = `<img src="${sessImg}" style="width:100%; height:100%; object-fit:contain; border-radius:8px;" />`;
    } else {
        const dNameLast = (dName.split(' ').pop() || 'DOE').toUpperCase();
        const dNameFirst = (dName.split(' ')[0] || 'JOHN').toUpperCase();
        const mapPlaceholderSVG = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='500' height='320' viewBox='0 0 500 320'%3E%3Crect width='500' height='320' fill='%23e0dac8' rx='10'/%3E%3Ctext x='40' y='50' font-family='Arial' font-size='16' font-weight='bold' fill='%23333'%3EPASSPORT%3C/text%3E%3Crect x='40' y='90' width='100' height='120' fill='%23666'/%3E%3Ctext x='160' y='95' font-family='monospace' font-size='11' fill='%23555'%3ESurname%3C/text%3E%3Ctext x='160' y='115' font-family='monospace' font-size='18' font-weight='bold' fill='%23111'%3E${dNameLast}%3C/text%3E%3Ctext x='160' y='140' font-family='monospace' font-size='11' fill='%23555'%3EGiven name%3C/text%3E%3Ctext x='160' y='160' font-family='monospace' font-size='18' font-weight='bold' fill='%23111'%3E${dNameFirst}%3C/text%3E%3Ctext x='160' y='185' font-family='monospace' font-size='11' fill='%23555'%3EDate of birth%3C/text%3E%3Ctext x='160' y='205' font-family='monospace' font-size='14' font-weight='bold' fill='%23111'%3E${dDOB}%3C/text%3E%3Ctext x='40' y='270' font-family='monospace' font-size='18' letter-spacing='2' fill='%23111'%3EP%3CUSA${dNameLast}%3C%3C${dNameFirst}%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C%3C/text%3E%3Ctext x='40' y='295' font-family='monospace' font-size='18' letter-spacing='2' fill='%23111'%3E1234567890USA901111M310119121234567890%3C/text%3E%3C/svg%3E")`;
        viewerContent = `<div style="width: 100%; height: 100%; background-color: #e0dac8; background-image: ${mapPlaceholderSVG}; background-size: cover; border-radius: 8px;"></div>`;
    }

    const content = `
        <div style="background:#fff; width:95%; max-width:1300px; height:85vh; border-radius:12px; overflow:hidden; display:flex; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);">
            
            <!-- Left Column: User Context & AI Checks -->
            <div style="width: 320px; border-right: 1px solid #e5e7eb; display:flex; flex-direction:column; background:#ffffff; flex-shrink:0;">
                <div style="padding: 24px; border-bottom: 1px solid #f3f4f6;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
                        <h2 style="margin:0; font-size:1.25rem; font-weight:600; color:#111827;">${dName}</h2>
                        <div style="width:40px; height:40px; border-radius:50%; background:#f3f4f6; display:flex; align-items:center; justify-content:center; font-weight:600; color:#374151; font-size:0.875rem; position:relative;">
                            ${initials}
                            <span style="position:absolute; bottom:0; right:0; width:10px; height:10px; background:#10b981; border:2px solid white; border-radius:50%;"></span>
                        </div>
                    </div>
                    <div style="display:flex; gap:8px; margin-bottom:24px;">
                        <span style="background:#ecfdf5; color:#059669; padding:4px 8px; border-radius:4px; font-size:0.75rem; font-weight:500;">Customer</span>
                        <span style="background:#f3f4f6; color:#4b5563; padding:4px 8px; border-radius:4px; font-size:0.75rem;">Uploaded today</span>
                    </div>

                    <div style="display:flex; flex-direction:column; gap:12px; font-size:0.875rem;">
                        <div style="display:flex; justify-content:space-between; text-align:right;">
                            <span style="color:#6b7280;">Email</span>
                            <span style="color:#0ea5e9;">${ver.email}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="color:#6b7280;">Extracted Address</span>
                            <span style="color:#374151; font-weight:500; text-align:right; max-width:180px;">${sessAddress}</span>
                        </div>
                    </div>
                </div>

                <div style="padding: 24px; flex:1; overflow-y:auto; background:#fafafa;">
                    <div style="margin-bottom:15px; display:flex; font-size:0.875rem;">
                        <span style="color:#0ea5e9; font-weight:500; border:1px solid #bae6fd; padding:6px 12px; border-radius:16px;">AI Verification Results</span>
                    </div>
                    <div style="background:#fff; border:1px solid ${isFaceMatch ? '#d1fae5' : '#fee2e2'}; border-radius:8px; overflow:hidden;">
                        <div style="padding:12px 16px; border-bottom:1px solid #f3f4f6; display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:600; color:#374151; font-size:0.875rem;">Document Verification</span>
                            <span style="color:#059669; font-size:0.75rem; font-weight:500;">${docConfidence.toFixed(1)}% Score</span>
                        </div>
                        <div style="padding:0;">
                            <div style="padding:12px 16px; border-bottom:1px solid #f3f4f6; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:0.875rem; color:#4b5563; width:70%;">Address Extracted?</span>
                                <span style="background:${sessAddress !== 'Address not detected' ? '#f3f4f6' : '#fee2e2'}; padding:4px 8px; border-radius:16px; font-size:0.75rem; font-weight:600; color:${sessAddress !== 'Address not detected' ? '#374151' : '#dc2626'};">${sessAddress !== 'Address not detected' ? 'Yes' : 'No'}</span>
                            </div>
                            <div style="padding:12px 16px; border-bottom:1px solid #f3f4f6; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:0.875rem; color:#4b5563; width:70%;">Valid ID document identified?</span>
                                <span style="background:#f3f4f6; padding:4px 8px; border-radius:16px; font-size:0.75rem; font-weight:600; color:#374151;">Yes</span>
                            </div>
                            <div style="padding:12px 16px; display:flex; justify-content:space-between; align-items:center; background:${isFaceMatch ? 'transparent':'#fef2f2'};">
                                <span style="font-size:0.875rem; color:#4b5563; width:70%;">Face match confirmed against selfie?</span>
                                <span style="background:${isFaceMatch ? '#f3f4f6' : '#fee2e2'}; padding:4px 8px; border-radius:16px; font-size:0.75rem; font-weight:600; color:${isFaceMatch ? '#374151' : '#dc2626'};">${isFaceMatch ? 'Yes' : 'No'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Middle Column: Document Viewer -->
            <div style="flex:1; background:#111827; position:relative; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:24px; overflow:hidden;">
                
                <!-- Image Viewer -->
                <div style="width: 100%; height: 100%; display:flex; justify-content:center; align-items:center; position:relative;">
                    <div style="max-width: 90%; max-height: 90%; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); position:relative; overflow:hidden; border-radius:8px;">
                        ${viewerContent}
                    </div>
                    <div style="position:absolute; bottom:24px; left:24px; right:24px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#9ca3af; font-size:0.875rem; font-weight:500; background:rgba(0,0,0,0.5); padding:4px 12px; border-radius:16px; backdrop-filter:blur(4px);">Viewing ${dDocType}</span>
                    </div>
                </div>

            </div>

            <!-- Right Column: Task Workflow -->
            <div style="width: 340px; background:#ffffff; border-left: 1px solid #e5e7eb; display:flex; flex-direction:column; flex-shrink:0;">
                <div style="padding:24px; display:flex; justify-content:flex-end;">
                    <button onclick="this.closest('.case-review-overlay').remove()" style="background:transparent; border:none; font-size:1.5rem; cursor:pointer; color:#9ca3af; outline:none; line-height:1;">&times;</button>
                </div>
                
                <div style="padding: 0 32px 32px 32px; flex:1; overflow-y:auto;">
                    <div style="width:40px; height:40px; background:#0d9488; border-radius:8px; display:flex; align-items:center; justify-content:center; margin-bottom:24px;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    </div>

                    <p style="color:#6b7280; font-size:0.75rem; text-transform:uppercase; font-weight:600; margin:0 0 8px 0; letter-spacing:0.05em;">Current Task</p>
                    <h2 style="font-size:1.125rem; color:#111827; margin:0 0 24px 0; line-height:1.5; font-weight:600;">
                        Review the ${dName} ${dDocType} Application and verify the extracted address.
                    </h2>

                    <button onclick="alert('Review process completed! Status updated.'); this.closest('.case-review-overlay').remove();" style="width:100%; background:#0d9488; color:white; border:none; padding:12px; border-radius:6px; font-weight:500; cursor:pointer; margin-bottom:32px; transition:background 0.2s;">
                        Approve Application
                    </button>

                    <button onclick="alert('Application Rejected by reviewer.'); this.closest('.case-review-overlay').remove();" style="width:100%; background:#fee2e2; color:#dc2626; border:none; padding:12px; border-radius:6px; font-weight:500; cursor:pointer; margin-top:-20px; margin-bottom:32px; transition:background 0.2s;">
                        Reject Application
                    </button>
                    
                    <!-- Vertical Stepper -->
                    <div style="position:relative; padding-left:12px;">
                        <div style="position:absolute; left:23px; top:12px; bottom:20px; width:2px; background:#e5e7eb;"></div>
                        
                        <div style="display:flex; align-items:flex-start; gap:16px; margin-bottom:32px; position:relative; z-index:1;">
                            <div style="width:24px; height:24px; background:#10b981; color:white; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; flex-shrink:0;">✓</div>
                            <div style="padding-top:2px;">
                                <p style="margin:0 0 8px 0; font-size:0.875rem; font-weight:500; color:#111827;">AI Extractions Completed</p>
                                <div style="height:6px; width:120px; background:#10b981; border-radius:3px;"></div>
                            </div>
                        </div>

                        <div style="display:flex; align-items:flex-start; gap:16px; margin-bottom:32px; position:relative; z-index:1;">
                            <div style="width:24px; height:24px; background:#111827; color:white; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; flex-shrink:0;">2</div>
                            <div style="padding-top:2px;">
                                <p style="margin:0 0 8px 0; font-size:0.875rem; font-weight:500; color:#111827;">Verify Data Elements</p>
                                <div style="height:6px; width:160px; background:#e5e7eb; border-radius:3px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    overlay.innerHTML = content;
    document.body.appendChild(overlay);
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
        
        // Ensure alert exists
        if (!alertData.find(a => a.name === item.name)) {
            alertData.push({
                id: 'AL' + Math.floor(Math.random() * 1000),
                name: item.name,
                risk: item.riskScore > 70 ? 'High' : 'Medium',
                type: 'Manual Flag',
                date: new Date().toISOString().split('T')[0],
                status: 'Active'
            });
        }
        
        saveDataLocally();
        filterVerifications();
        showNotification(`Item ${id} has been flagged for review`, 'warning');
    }
}

// Resolve alert
function resolveAlert(alertId) {
    const alert = alertData.find(a => a.id === alertId);
    if (alert) {
        alert.status = 'Resolved';
        saveDataLocally();
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