from config import APP_NAME, APP_VERSION, DEBUG, OPENAI_API_KEY


def main() -> None:
    print(f"Starting {APP_NAME} v{APP_VERSION}  (debug={DEBUG})")

    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")


if __name__ == "__main__":
    main()
