        if not self._using_codex_oauth:
            fb_str = os.environ.get("OUROBOROS_MODEL_FALLBACK_LIST", "") or os.environ.get("OUROBOROS_MODEL_FALLBACKS", "")
            if fb_str:
                self._fallback_chain = [m.strip() for m in fb_str.split(",") if m.strip()]
            else:
                # Sensible default fallback chain: free models across providers
                self._fallback_chain = [
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "google/gemini-2.0-flash-001:free",
                    "mistralai/mistral-7b-instruct:free",
                ]