const ClaimAPI = {

    baseUrl: "/api/claims",

    async request(path, options = {}) {
        const response = await fetch(
            `${this.baseUrl}${path}`,
            options
        );

        if (!response.ok) {
            const message = await response.text();

            throw new Error(
                `API error ${response.status}: ${message}`
            );
        }

        return response.json();
    },

    // NEXT
    async next() {
        return this.request("/next");
    },

    // ANALYZE
    async analyze(claimId) {
        return this.request(
            `/${encodeURIComponent(claimId)}/analyze`,
            { method: "POST" }
        );
    },

    // CONFIRM
    async confirm(claimId, humanScore) {
        return this.request(
            `/${encodeURIComponent(claimId)}/confirm`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ humanScore })
            }
        );
    },

    // REVEAL
    async reveal(claimId) {
        return this.request(
            `/${encodeURIComponent(claimId)}/reveal`
        );
    }
};
