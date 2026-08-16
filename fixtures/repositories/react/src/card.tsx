export interface CardProps { title: string; tags?: readonly string[] }
export function Card({ title, tags = [] }: CardProps) {
  return <article><h2>{title}</h2>{tags.map((tag) => <span key={tag}>{tag}</span>)}</article>;
}
