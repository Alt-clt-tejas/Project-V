# project_oracle/test_import.py

print("--- Starting Diagnostic Import ---")

try:
    print("Attempting to import 'app.config.base'...")
    from app.config import base
    print("[SUCCESS] Imported 'app.config.base'")

    print("\nAttempting to import 'app.domains.search.schemas'...")
    from app.domains.search import schemas
    print("[SUCCESS] Imported 'app.domains.search.schemas'")

    print("\nAttempting to import 'app.connectors.base_connector'...")
    from app.connectors import base_connector
    print("[SUCCESS] Imported 'app.connectors.base_connector'")

    print("\nAttempting to import 'app.connectors.youtube_connector'...")
    from app.connectors import youtube_connector
    print("[SUCCESS] Imported 'app.connectors.youtube_connector'")

    print("\nAttempting to import 'app.domains.search.service'...")
    from app.domains.search import service
    print("[SUCCESS] Imported 'app.domains.search.service'")

    print("\nAttempting to import 'app.agents.s5_search_agent'...")
    from app.agents import s5_search_agent
    print("[SUCCESS] Imported 'app.agents.s5_search_agent'")

    print("\nAttempting to import 'app.api.dependencies'...")
    from app.api import dependencies
    print("[SUCCESS] Imported 'app.api.dependencies'")

    print("\nAttempting to import 'app.api.v1.routes.search_routes'...")
    from app.api import search
    print("[SUCCESS] Imported 'app.api.v1.routes.search_routes'")

    print("\nAttempting to import FINAL MODULE 'app.main'...")
    import app.main
    print("[SUCCESS] Imported 'app.main'")

    print("\n--- Diagnostic Complete: All imports were successful. ---")

except Exception as e:
    print(f"\n--- !!! IMPORT FAILED !!! ---")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR DETAILS: {e}")
    print("\nThe error occurred in the last module listed above.")
    import traceback
    traceback.print_exc()