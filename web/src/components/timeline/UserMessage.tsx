import type { UserItem } from "@/store/timeline";

export function UserMessage({ item }: { item: UserItem }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-muted px-4 py-2.5 text-sm whitespace-pre-wrap break-words">
        {item.text}
      </div>
    </div>
  );
}
